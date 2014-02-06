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
  , m_do_mask()
  , m_do_bkgd()
  , m_do_gain()
  , m_do_nrms()
  , m_do_thre() 
  , m_fname_peds()
  , m_fname_bkgd()
  , m_fname_gain()
  , m_fname_mask()
  , m_fname_nrms()
  , m_mask_val()
  , m_low_nrms()
  , m_low_thre()
  , m_low_val()
  , m_row_min()
  , m_row_max()
  , m_col_min()
  , m_col_max()
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
  m_do_mask           = config   ("do_mask",              false );
  m_do_bkgd           = config   ("do_bkgd",              false );
  m_do_gain           = config   ("do_gain",              false );
  m_do_nrms           = config   ("do_nrms",              false );
  m_do_thre           = config   ("do_thre",              false );
  m_fname_peds        = configStr("fname_peds",               "");
  m_fname_bkgd        = configStr("fname_bkgd",               "");
  m_fname_gain        = configStr("fname_gain",               "");
  m_fname_mask        = configStr("fname_mask",               "");
  m_fname_nrms        = configStr("fname_rms",                "");
  m_mask_val          = config   ("masked_value",             0.);
  m_low_nrms          = config   ("threshold_nrms",           3.);
  m_low_thre          = config   ("threshold",                0.);
  m_low_val           = config   ("below_thre_value",         0.);
  m_row_min           = config   ("bkgd_row_min",             0 );
  m_row_max           = config   ("bkgd_row_max",            10 );
  m_col_min           = config   ("bkgd_col_min",             0 );
  m_col_max           = config   ("bkgd_col_max",            10 );
  m_print_bits        = config   ("print_bits",               0 );

  /*
  m_do_peds = (m_fname_peds.empty()) ? false : true;
  m_do_mask = (m_fname_mask.empty()) ? false : true;
  m_do_bkgd = (m_fname_bkgd.empty()) ? false : true;
  m_do_gain = (m_fname_gain.empty()) ? false : true;
  m_do_nrms = (m_fname_nrms.empty()) ? false : true;
  */

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
        << "\n m_do_mask         : " << m_do_mask     
        << "\n m_do_bkgd         : " << m_do_bkgd     
        << "\n m_do_gain         : " << m_do_gain     
        << "\n m_do_nrms         : " << m_do_nrms     
        << "\n m_do_thre         : " << m_do_thre     
        << "\n m_fname_peds      : " << m_fname_peds
        << "\n m_fname_mask      : " << m_fname_mask     
        << "\n m_fname_bkgd      : " << m_fname_bkgd     
        << "\n m_fname_gain      : " << m_fname_gain     
        << "\n m_fname_nrms      : " << m_fname_nrms     
        << "\n m_mask_val        : " << m_mask_val   
        << "\n m_low_nrms        : " << m_low_nrms   
        << "\n m_low_thre        : " << m_low_thre   
        << "\n m_low_val         : " << m_low_val   
        << "\n m_row_min         : " << m_row_min    
        << "\n m_row_max         : " << m_row_max    
        << "\n m_col_min         : " << m_col_min    
        << "\n m_col_max         : " << m_col_max    
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
  //getConfigPars(env);      // get m_src here
  //getCalibPars(evt, env);  // use m_src here
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
  //if(!m_count_event) init(evt, env);
  if( m_print_bits & 2 ) printEventRecord(evt);
  procEvent(evt, env);
  // saveImageInEvent(evt); -> moved to procEventForType
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
NDArrCalib::init(Event& evt, Env& env)
{
    defineImageShape(evt, m_str_src, m_key_in, m_shape); // shape is not available in beginJob and beginRun
    m_rows = m_shape[0];
    m_cols = m_shape[1];
    m_size = m_rows*m_cols;

    if( m_do_peds ) m_peds = new ImgAlgos::ImgParametersV1(m_fname_peds);
    else            m_peds = new ImgAlgos::ImgParametersV1(m_shape);   // zero array

    if( m_do_mask ) m_mask = new ImgAlgos::ImgParametersV1(m_fname_mask);
    else            m_mask = new ImgAlgos::ImgParametersV1(m_shape,1); // unit array

    if( m_do_bkgd ) m_bkgd = new ImgAlgos::ImgParametersV1(m_fname_bkgd);
    else            m_bkgd = new ImgAlgos::ImgParametersV1(m_shape);   // zero array

    if( m_do_gain ) m_gain = new ImgAlgos::ImgParametersV1(m_fname_gain);
    else            m_gain = new ImgAlgos::ImgParametersV1(m_shape,1); // unit array

    if( m_do_nrms ) m_nrms = new ImgAlgos::ImgParametersV1(m_fname_nrms, m_low_nrms); // load rms array with factor
    else            m_nrms = new ImgAlgos::ImgParametersV1(m_shape);   // zero array

    m_peds_data = m_peds->data();
    m_bkgd_data = m_bkgd->data();
    m_gain_data = m_gain->data();
    m_mask_data = m_mask->data();
    m_nrms_data = m_nrms->data();

    if( m_do_peds && m_print_bits & 4 ) m_peds -> print("Pedestals");
    if( m_do_mask && m_print_bits & 4 ) m_mask -> print("Mask");
    if( m_do_bkgd && m_print_bits & 4 ) m_bkgd -> print("Background");
    if( m_do_gain && m_print_bits & 4 ) m_gain -> print("Gain");
    if( m_do_nrms && m_print_bits & 4 ) m_nrms -> print("RMS threshold");

    //m_cdat = new double [m_size];
    //std::fill_n(m_cdat, int(m_size), 0);    

    if( m_do_bkgd ) defImgIndexesForBkgdNorm();
}
*/

//--------------------

void 
NDArrCalib::getConfigPars(Env& env)
{
  //  m_count_cfg = 0; 
  //if ( getConfigParsForType <Psana::CsPad2x2::ConfigV1> (env) ) return;


  if ( getConfigParsForType <Psana::PNCCD::ConfigV1> (env) ) return;
  if ( getConfigParsForType <Psana::PNCCD::ConfigV2> (env) ) return;

  MsgLog(name(), error, "No configuration objects found, terminating.");
  terminate();
}

//--------------------

void 
NDArrCalib::getCalibPars(Event& evt, Env& env)
{
  //std::string calib_dir = (m_calibDir == "") ? env.calibDir() : m_calibDir;
  std::string calib_dir = env.calibDir();

  MsgLog(name(), info, "Calibration directory: " << calib_dir);
  m_typeGroup  = std::string(); // "PNCCD::CalibV1";
  m_calibpars = PSCalib::CalibParsStore::Create(calib_dir, m_typeGroup, m_src, getRunNumber(evt));

  if( m_print_bits & 4 ) m_calibpars->printCalibPars();

  m_peds_data = m_calibpars->pedestals();
  m_gain_data = m_calibpars->pixel_gain();
  m_mask_data = m_calibpars->pixel_status();
  m_cmod_data = m_calibpars->common_mode();

  // TEMPORARY SOLUTION: 
  // PER PIXEL BACKGROUND PARAMETERS = 0
  // PER PIXEL THRESHOLD PARAMETERS = m_low_thre

  m_bkgd_data = new PSCalib::CalibPars::pixel_bkgd_t[m_size];
  m_nrms_data = new PSCalib::CalibPars::pixel_nrms_t[m_size];

  std::fill_n(m_bkgd_data, int(m_size), PSCalib::CalibPars::pixel_bkgd_t(0));    
  std::fill_n(m_nrms_data, int(m_size), PSCalib::CalibPars::pixel_nrms_t(m_low_thre));    

  /*
  if( m_print_bits & 2 ) {
    m_calibpars  -> printInputPars();
    m_calibpars  -> printCalibPars();
    //m_pix_coords_cspad2x2 -> printCoordArray(); 
    //m_pix_coords_cspad2x2 -> printConstants(); 
  }
  */
}

//--------------------

void
NDArrCalib::initAtFirstGetNdarray(Event& evt, Env& env)
{
  if ( !defineNDArrPars(evt, m_str_src, m_key_in, m_ndarr_pars) ) return;

  // Do initialization for known ndarray type and NDims
  if( m_print_bits & 1 ) m_ndarr_pars -> print();
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
  for(unsigned r = m_row_min; r < m_row_max+1; r++) {
    for(unsigned c = m_col_min; c < m_col_max+1; c++) 
      v_inds.push_back(r*m_cols+c);
  }
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
//--------------------
} // namespace ImgAlgos
