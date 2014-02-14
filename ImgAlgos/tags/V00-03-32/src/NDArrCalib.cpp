//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NDArrCalib...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/NDArrCalib.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
#include "PSCalib/CalibParsStore.h"
#include "pdscalibdata/GlobalMethods.h" // for load_pars_from_file(...)

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(NDArrCalib)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
NDArrCalib::NDArrCalib (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out() 
  , m_do_peds()
  , m_do_cmod()
  , m_do_stat()
  , m_do_mask()
  , m_do_bkgd()
  , m_do_gain()
  , m_do_nrms()
  , m_do_thre() 
  , m_fname_mask()
  , m_fname_bkgd()
  , m_mask_val()
  , m_low_nrms()
  , m_low_thre()
  , m_low_val()
  , m_ind_min()
  , m_ind_max()
  , m_ind_inc()
  , m_print_bits()
  , m_count_event(0)
  , m_count_get(0)
{
  // get the values from configuration or use defaults
  m_str_src           = configSrc("source",   "DetInfo(:Camera)");
  m_key_in            = configStr("key_in",                   "");
  m_key_out           = configStr("key_out",        "calibrated");
  m_do_peds           = config   ("do_peds",              false );
  m_do_cmod           = config   ("do_cmod",              false );
  m_do_stat           = config   ("do_stat",              false );
  m_do_mask           = config   ("do_mask",              false );
  m_do_bkgd           = config   ("do_bkgd",              false );
  m_do_gain           = config   ("do_gain",              false );
  m_do_nrms           = config   ("do_nrms",              false );
  m_do_thre           = config   ("do_thre",              false );
  m_fname_mask        = configStr("fname_mask",               "");
  m_fname_bkgd        = configStr("fname_bkgd",               "");
  m_mask_val          = config   ("masked_value",             0.);
  m_low_nrms          = config   ("threshold_nrms",           3.);
  m_low_thre          = config   ("threshold",                0.);
  m_low_val           = config   ("below_thre_value",         0.);
  m_ind_min           = config   ("bkgd_ind_min",             0 );
  m_ind_max           = config   ("bkgd_ind_max",           100 );
  m_ind_inc           = config   ("bkgd_ind_inc",             2 );
  m_print_bits        = config   ("print_bits",               0 );

  m_ndarr_pars = new NDArrPars();
}

//--------------------

void 
NDArrCalib::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters  :"
        << "\n source            : " << m_str_src
        << "\n m_key_in          : " << m_key_in      
        << "\n m_key_out         : " << m_key_out
        << "\n m_do_peds         : " << m_do_peds
        << "\n m_do_cmod         : " << m_do_cmod
        << "\n m_do_stat         : " << m_do_stat
        << "\n m_do_mask         : " << m_do_mask     
        << "\n m_do_bkgd         : " << m_do_bkgd     
        << "\n m_do_gain         : " << m_do_gain     
        << "\n m_do_nrms         : " << m_do_nrms     
        << "\n m_do_thre         : " << m_do_thre     
        << "\n m_fname_mask      : " << m_fname_mask     
        << "\n m_fname_bkgd      : " << m_fname_bkgd     
        << "\n m_mask_val        : " << m_mask_val   
        << "\n m_low_nrms        : " << m_low_nrms   
        << "\n m_low_thre        : " << m_low_thre   
        << "\n m_low_val         : " << m_low_val   
        << "\n m_ind_min         : " << m_ind_min    
        << "\n m_ind_max         : " << m_ind_max    
        << "\n m_ind_inc         : " << m_ind_inc
        << "\n m_print_bits      : " << m_print_bits
	<< "\n Output data type  : " << typeid(data_out_t).name() << " of size " << sizeof(data_out_t)
        << "\n";     

    printSizeOfTypes();
  }
}

//--------------------


//--------------
// Destructor --
//--------------
NDArrCalib::~NDArrCalib ()
{
}

/// Method which is called once at the beginning of the job
void 
NDArrCalib::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

/// Method which is called at the beginning of the run
void 
NDArrCalib::beginRun(Event& evt, Env& env)
{
  m_count_get = 0; // In order to load calibration pars etc for each run
}

/// Method which is called at the beginning of the calibration cycle
void 
NDArrCalib::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
NDArrCalib::event(Event& evt, Env& env)
{
  if( m_print_bits & 16 ) printEventRecord(evt);
  procEvent(evt, env);
  ++ m_count_event;
}
  
/// Method which is called at the end of the calibration cycle
void 
NDArrCalib::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
NDArrCalib::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
NDArrCalib::endJob(Event& evt, Env& env)
{
}

//--------------------

/*
void 
NDArrCalib::getConfigPars(Env& env)
{
  //if ( getConfigParsForType <Psana::CsPad2x2::ConfigV1> (env) ) return;
  if ( getConfigParsForType <Psana::PNCCD::ConfigV1> (env) ) return;
  if ( getConfigParsForType <Psana::PNCCD::ConfigV2> (env) ) return;
  MsgLog(name(), error, "No configuration objects found, terminating.");
  terminate();
}
*/

//--------------------

void 
NDArrCalib::getCalibPars(Event& evt, Env& env)
{
  //std::string calib_dir = (m_calibDir == "") ? env.calibDir() : m_calibDir;
  std::string calib_dir = env.calibDir();

  MsgLog(name(), info, "Calibration directory: " << calib_dir);
  std::string m_group = std::string(); // for ex: "PNCCD::CalibV1";
  m_calibpars = PSCalib::CalibParsStore::Create(calib_dir, m_group, m_src, getRunNumber(evt));

  if( m_print_bits & 2 ) m_calibpars->printCalibPars();

  m_peds_data = m_calibpars->pedestals();
  m_gain_data = m_calibpars->pixel_gain();
  m_stat_data = m_calibpars->pixel_status();
  m_cmod_data = m_calibpars->common_mode();
  m_rms_data  = m_calibpars->pixel_rms();

  static bool is_done_once = false; 
  if ( ! is_done_once ) {
         is_done_once = true;
      m_bkgd_data = new PSCalib::CalibPars::pixel_bkgd_t[m_size];
      m_mask_data = new PSCalib::CalibPars::pixel_mask_t[m_size];
      m_nrms_data = new data_out_t[m_size];

      if( m_do_mask && !m_fname_mask.empty())
        pdscalibdata::load_pars_from_file<PSCalib::CalibPars::pixel_mask_t> (m_fname_mask, "Mask", m_size, m_mask_data);
      else std::fill_n(m_mask_data, int(m_size), PSCalib::CalibPars::pixel_mask_t(1));    
      
      if( m_do_bkgd && !m_fname_bkgd.empty())
        pdscalibdata::load_pars_from_file<PSCalib::CalibPars::pixel_bkgd_t> (m_fname_bkgd, "Background", m_size, m_bkgd_data);
      else std::fill_n(m_bkgd_data, int(m_size), PSCalib::CalibPars::pixel_bkgd_t(0));    
  }

  if( m_do_nrms ) { for(unsigned i=0; i<m_size; i++) m_nrms_data[i] = m_low_nrms * m_rms_data[i]; }

  if( m_print_bits & 4 ) printCommonModePars();
}

//--------------------

void
NDArrCalib::initAtFirstGetNdarray(Event& evt, Env& env)
{
  if ( !defineNDArrPars(evt, m_str_src, m_key_in, m_ndarr_pars) ) return;

  // Do initialization for known ndarray type and NDims
  if( m_print_bits & 8 ) m_ndarr_pars -> print();
  m_size  = m_ndarr_pars->size();
  m_dtype = m_ndarr_pars->dtype();
  m_ndim  = m_ndarr_pars->ndim();
  m_src   = m_ndarr_pars->src();

  p_cdata = new data_out_t[m_size];

  getCalibPars(evt, env);
}
 
//--------------------

void 
NDArrCalib::procEvent(Event& evt, Env& env)
{
  if ( ! m_count_get ) initAtFirstGetNdarray(evt, env);
  if ( ! m_ndarr_pars -> is_set() ) return;

  if      (m_dtype == UINT16   && procEventForType<uint16_t, data_out_t> (evt) ) return;
  else if (m_dtype == INT      && procEventForType<int,      data_out_t> (evt) ) return;
  else if (m_dtype == FLOAT    && procEventForType<float,    data_out_t> (evt) ) return;
  else if (m_dtype == UINT8    && procEventForType<uint8_t,  data_out_t> (evt) ) return;
  else if (m_dtype == DOUBLE   && procEventForType<double,   data_out_t> (evt) ) return;

  MsgLog(name(), info, "Image is not available in the event(...) for source:" << m_str_src << " key:" << m_key_in);
}

//--------------------
 
void 
NDArrCalib::defImgIndexesForBkgdNorm()
{
  v_inds.clear(); 
  for(unsigned i = m_ind_min; i < m_ind_max; i+=m_ind_inc) 
      v_inds.push_back(i);
}

//--------------------

void 
NDArrCalib::printEventRecord(Event& evt)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count_event) 
                     << " get="    << stringFromUint(m_count_get) 
                     << " Time="   << stringTimeStamp(evt) 
  );
}

//--------------------

void 
NDArrCalib::printCommonModePars()
{
     std::stringstream ss; ss << "Common mode parameters: "; 
     for (int i=0; i<16;  ++i) ss << " " << m_cmod_data[i];
     MsgLog( name(), info, ss.str());
}

//--------------------
} // namespace ImgAlgos
//--------------------
