//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgCalib...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgCalib.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ImgCalib)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
ImgCalib::ImgCalib (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out() 
  , m_fname_peds()
  , m_fname_bkgd()
  , m_fname_gain()
  , m_fname_mask()
  , m_fname_nrms()
  , m_mask_val()
  , m_low_nrms()
  , m_low_thre()
  , m_low_val()
  , m_do_thre() 
  , m_row_min()
  , m_row_max()
  , m_col_min()
  , m_col_max()
  , m_print_bits()
  , m_count_event(0)
  , m_count_get(0)
  , m_count_msg(0)
{
  // get the values from configuration or use defaults
  m_str_src           = configSrc("source",   "DetInfo(:Camera)");
  m_key_in            = configStr("key_in",                   "");
  m_key_out           = configStr("key_out",        "calibrated");
  m_fname_peds        = configStr("fname_peds",               "");
  m_fname_bkgd        = configStr("fname_bkgd",               "");
  m_fname_gain        = configStr("fname_gain",               "");
  m_fname_mask        = configStr("fname_mask",               "");
  m_fname_nrms        = configStr("fname_rms",                "");
  m_mask_val          = config   ("masked_value",             0.);
  m_low_nrms          = config   ("threshold_nrms",           3.);
  m_low_thre          = config   ("threshold",                0.);
  m_low_val           = config   ("below_thre_value",         0.);
  m_do_thre           = config   ("do_threshold",         false );
  m_row_min           = config   ("bkgd_row_min",             0 );
  m_row_max           = config   ("bkgd_row_max",            10 );
  m_col_min           = config   ("bkgd_col_min",             0 );
  m_col_max           = config   ("bkgd_col_max",            10 );
  m_print_bits        = config   ("print_bits",               0 );

  m_do_peds = (m_fname_peds.empty()) ? false : true;
  m_do_mask = (m_fname_mask.empty()) ? false : true;
  m_do_bkgd = (m_fname_bkgd.empty()) ? false : true;
  m_do_gain = (m_fname_gain.empty()) ? false : true;
  m_do_nrms = (m_fname_nrms.empty()) ? false : true;
}

//--------------------

void 
ImgCalib::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters  :"
        << "\n source            : " << m_str_src
        << "\n m_key_in          : " << m_key_in      
        << "\n m_key_out         : " << m_key_out
        << "\n m_fname_peds      : " << m_fname_peds
        << "\n m_fname_mask      : " << m_fname_mask     
        << "\n m_fname_bkgd      : " << m_fname_bkgd     
        << "\n m_fname_gain      : " << m_fname_gain     
        << "\n m_fname_nrms      : " << m_fname_nrms     
        << "\n m_do_peds         : " << m_do_peds
        << "\n m_do_mask         : " << m_do_mask     
        << "\n m_do_bkgd         : " << m_do_bkgd     
        << "\n m_do_gain         : " << m_do_gain     
        << "\n m_do_nrms         : " << m_do_nrms     
        << "\n m_do_thre         : " << m_do_thre     
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
ImgCalib::~ImgCalib ()
{
}

/// Method which is called once at the beginning of the job
void 
ImgCalib::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();

}

/// Method which is called at the beginning of the run
void 
ImgCalib::beginRun(Event& evt, Env& env)
{
  m_count_get = 0; // In order to load calibration pars etc for each run
}

/// Method which is called at the beginning of the calibration cycle
void 
ImgCalib::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgCalib::event(Event& evt, Env& env)
{
  ++ m_count_event;
  if( m_print_bits & 2 ) printEventRecord(evt);
  procEvent(evt, env);
  // saveImageInEvent(evt); -> moved to procEventForType
}
  
/// Method which is called at the end of the calibration cycle
void 
ImgCalib::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ImgCalib::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ImgCalib::endJob(Event& evt, Env& env)
{
}

//--------------------

void 
ImgCalib::init(Event& evt, Env& env)
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

//--------------------

void 
ImgCalib::procEvent(Event& evt, Env& env)
{
  if ( ! m_count_get  ) init(evt, env);

  if ( procEventForType<int16_t,  data_out_t> (evt) ) return;
  if ( procEventForType<uint16_t, data_out_t> (evt) ) return;
  if ( procEventForType<int,      data_out_t> (evt) ) return;
  if ( procEventForType<float,    data_out_t> (evt) ) return;
  if ( procEventForType<double,   data_out_t> (evt) ) return;
  if ( procEventForType<uint8_t,  data_out_t> (evt) ) return;

  if (++m_count_msg < 11 && m_print_bits) {
    MsgLog(name(), warning, "Image is not available in the event:" << m_count_event << " for source:" << m_str_src << " key:" << m_key_in);
    if (m_count_msg == 10) MsgLog(name(), warning, "STOP WARNINGS for source:" << m_str_src << " key:" << m_key_in);    
  }
}

//--------------------
 
void 
ImgCalib::defImgIndexesForBkgdNorm()
{
  v_inds.clear();
  for(unsigned r = m_row_min; r < m_row_max+1; r++) {
    for(unsigned c = m_col_min; c < m_col_max+1; c++) 
      v_inds.push_back(r*m_cols+c);
  }
}

//--------------------

void 
ImgCalib::printEventRecord(Event& evt)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " evt="    << stringFromUint(m_count_event) 
                     << " get="    << stringFromUint(m_count_get) 
                     << " time="   << stringTimeStamp(evt) 
  );
}

//--------------------
//--------------------
} // namespace ImgAlgos
