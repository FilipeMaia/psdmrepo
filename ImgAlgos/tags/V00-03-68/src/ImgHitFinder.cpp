//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgHitFinder...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgHitFinder.h"

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
PSANA_MODULE_FACTORY(ImgHitFinder)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
ImgHitFinder::ImgHitFinder (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out() 
  , m_fname_peds()
  , m_fname_mask()
  , m_fname_gain()
  , m_fname_thre()
  , m_masked_val()
  , m_thre_mode()
  , m_thre_param()
  , m_thre_below_val()
  , m_thre_above_val()
  , m_row_min()
  , m_row_max()
  , m_col_min()
  , m_col_max()
  , m_print_bits()
  , m_count(0)
{
  m_str_src           = configSrc("source",         "DetInfo(:Camera)");
  m_key_in            = configStr("key_in",                         "");
  m_key_out           = configStr("key_out",                "img-hits");
  m_fname_peds        = configStr("fname_peds",                     "");
  m_fname_mask        = configStr("fname_mask",                     "");
  m_fname_gain        = configStr("fname_gain",                     "");
  m_fname_thre        = configStr("fname_thre",                     "");
  m_masked_val        = config   ("masked_value",                   0.);
  m_thre_mode         = config   ("thre_mode",                      1 );
  m_thre_param        = config   ("thre_param",           k_val_def_d );
  m_thre_below_val    = config   ("thre_below_value",               0.);
  m_thre_above_val    = config   ("thre_above_value",     k_val_def_d );
  m_row_min           = config   ("win_row_min",                    1 );
  m_row_max           = config   ("win_row_max",          k_val_def_u );
  m_col_min           = config   ("win_col_min",                    1 );
  m_col_max           = config   ("win_col_max",          k_val_def_u );
  m_print_bits        = config   ("print_bits",                     0 );

  m_do_peds = (m_fname_peds.empty()) ? false : true;
  m_do_mask = (m_fname_mask.empty()) ? false : true;
  m_do_gain = (m_fname_gain.empty()) ? false : true;
  m_do_thre = (m_thre_mode == 0) ? false : true;
}

//--------------------

void 
ImgHitFinder::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters  :"
        << "\n source            : " << m_str_src
        << "\n m_key_in          : " << m_key_in      
        << "\n m_key_out         : " << m_key_out
        << "\n m_fname_peds      : " << m_fname_peds
        << "\n m_fname_mask      : " << m_fname_mask     
        << "\n m_fname_gain      : " << m_fname_gain     
        << "\n m_fname_thre      : " << m_fname_thre     
        << "\n m_do_peds         : " << m_do_peds
        << "\n m_do_mask         : " << m_do_mask     
        << "\n m_do_gain         : " << m_do_gain     
        << "\n m_do_thre         : " << m_do_thre     
        << "\n m_masked_val      : " << m_masked_val   
        << "\n m_thre_mode       : " << m_thre_mode   
        << "\n m_thre_param      : " << m_thre_param   
        << "\n m_thre_below_val  : " << m_thre_below_val   
        << "\n m_thre_above_val  : " << m_thre_above_val   
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
ImgHitFinder::~ImgHitFinder ()
{
}

/// Method which is called once at the beginning of the job
void 
ImgHitFinder::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
ImgHitFinder::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
ImgHitFinder::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgHitFinder::event(Event& evt, Env& env)
{
  if(!m_count) init(evt, env);
  if( m_print_bits & 2 ) printEventRecord(evt);
  procEvent(evt, env);
  // saveImageInEvent(evt); -> moved to procEventForType
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
ImgHitFinder::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ImgHitFinder::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ImgHitFinder::endJob(Event& evt, Env& env)
{
}

//--------------------

void 
ImgHitFinder::init(Event& evt, Env& env)
{
    defineImageShape(evt, m_str_src, m_key_in, m_shape); // shape is not available in beginJob and beginRun
    m_rows = m_shape[0];
    m_cols = m_shape[1];
    m_size = m_rows*m_cols;

    if (m_row_max == k_val_def_u ) m_row_max = m_rows-1;
    if (m_col_max == k_val_def_u ) m_col_max = m_cols-1;

    m_peds = ( m_fname_peds.empty() ) ? new ImgAlgos::ImgParametersV1(m_shape)       // zero array
                                      : new ImgAlgos::ImgParametersV1(m_fname_peds);

    m_mask = ( m_fname_mask.empty() ) ? new ImgAlgos::ImgParametersV1(m_shape, 1)    // unit array
                                      : new ImgAlgos::ImgParametersV1(m_fname_mask); 

    m_gain = ( m_fname_gain.empty() ) ? new ImgAlgos::ImgParametersV1(m_shape, 1)    // unit array
                                      : new ImgAlgos::ImgParametersV1(m_fname_gain); 

    m_thre = ( m_fname_thre.empty() ) ? new ImgAlgos::ImgParametersV1(m_shape)       // zero array
                                      : new ImgAlgos::ImgParametersV1(m_fname_thre, m_thre_param); // load constant threshold 

    // get pointers to the objects' content
    m_peds_data = m_peds->data();
    m_mask_data = m_mask->data();
    m_gain_data = m_gain->data();
    m_thre_data = m_thre->data();

    // make ndarrays from the objects' content
    m_peds_nda = make_ndarray(m_peds_data, m_rows, m_cols);
    m_mask_nda = make_ndarray(m_mask_data, m_rows, m_cols);
    m_gain_nda = make_ndarray(m_gain_data, m_rows, m_cols);
    m_thre_nda = make_ndarray(m_thre_data, m_rows, m_cols);

    if( m_do_peds && m_print_bits & 4 ) m_peds -> print("Pedestals");
    if( m_do_mask && m_print_bits & 4 ) m_mask -> print("Mask");
    if( m_do_gain && m_print_bits & 4 ) m_gain -> print("Gain");
    if( m_do_thre && m_print_bits & 4 ) m_thre -> print("Threshold");

    //m_cdat = new double [m_size];
    //std::fill_n(m_cdat, int(m_size), 0);    

    //if( m_do_bkgd ) defImgIndexesForBkgdNorm();

    if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

void 
ImgHitFinder::procEvent(Event& evt, Env& env)
{
  if ( procEventForType<uint8_t,  data_out_t> (evt) ) return;
  if ( procEventForType<uint16_t, data_out_t> (evt) ) return;
  if ( procEventForType<int,      data_out_t> (evt) ) return;
  if ( procEventForType<float,    data_out_t> (evt) ) return;
  if ( procEventForType<double,   data_out_t> (evt) ) return;

  MsgLog(name(), info, "Image is not available in the event(...) for source:" << m_str_src << " key:" << m_key_in);
}

//--------------------

void 
ImgHitFinder::printEventRecord(Event& evt)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
  );
}

//--------------------
//--------------------
/* 
void 
ImgHitFinder::defImgIndexesForBkgdNorm()
{
  v_inds.clear();
  for(unsigned r = m_row_min; r < m_row_max+1; r++) {
    for(unsigned c = m_col_min; c < m_col_max+1; c++) 
      v_inds.push_back(r*m_cols+c);
  }
}
*/
//--------------------
} // namespace ImgAlgos
