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
#include "ImgAlgos/GlobalMethods.h"


//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ImgCalib)

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
  , m_mask_val()
  , m_row_min()
  , m_row_max()
  , m_col_min()
  , m_col_max()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src           = configStr("source", "DetInfo(:Camera)");
  m_key_in            = configStr("key_in",                 "");
  m_key_out           = configStr("key_out",      "calibrated");
  m_fname_peds        = configStr("fname_peds",             "");
  m_fname_bkgd        = configStr("fname_bkgd",             "");
  m_fname_gain        = configStr("fname_gain",             "");
  m_fname_mask        = configStr("fname_mask",             "");
  m_mask_val          = config   ("masked_value",           0 );
  m_row_min           = config   ("bkgd_row_min",           0 );
  m_row_max           = config   ("bkgd_row_max",          10 );
  m_col_min           = config   ("bkgd_col_min",           0 );
  m_col_max           = config   ("bkgd_col_max",          10 );
  m_print_bits        = config   ("print_bits",             0 );

  m_do_peds = (m_fname_peds.empty()) ? false : true;
  m_do_mask = (m_fname_mask.empty()) ? false : true;
  m_do_bkgd = (m_fname_bkgd.empty()) ? false : true;
  m_do_gain = (m_fname_gain.empty()) ? false : true;
}

//--------------------

void 
ImgCalib::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters  :"
        << "\n source            : " << m_str_src
        << "\n key_in            : " << m_key_in      
        << "\n key_out           : " << m_key_out
        << "\n m_fname_peds      : " << m_fname_peds
        << "\n m_fname_mask      : " << m_fname_mask     
        << "\n m_fname_bkgd      : " << m_fname_bkgd     
        << "\n m_fname_gain      : " << m_fname_gain     
        << "\n m_do_peds         : " << m_do_peds
        << "\n m_do_mask         : " << m_do_mask     
        << "\n m_do_bkgd         : " << m_do_bkgd     
        << "\n m_do_gain         : " << m_do_gain     
        << "\n m_mask_val        : " << m_mask_val   
        << "\n m_row_min         : " << m_row_min    
        << "\n m_row_max         : " << m_row_max    
        << "\n m_col_min         : " << m_col_min    
        << "\n m_col_max         : " << m_col_max    
        << "\n print_bits        : " << m_print_bits
        << "\n";     
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
  if(!m_count) init(evt, env);
  if( m_print_bits & 2 ) printEventRecord(evt);
  procEvent(evt, env);
  saveImageInEvent(evt);
  ++ m_count;
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
    defineImageShape(evt); // shape is not available in beginJob and beginRun

    if( m_do_peds ) m_peds = new ImgAlgos::ImgParametersV1(m_fname_peds);
    else            m_peds = new ImgAlgos::ImgParametersV1(m_shape);   // zero array

    if( m_do_mask ) m_mask = new ImgAlgos::ImgParametersV1(m_fname_mask);
    else            m_mask = new ImgAlgos::ImgParametersV1(m_shape,1); // unit array

    if( m_do_bkgd ) m_bkgd = new ImgAlgos::ImgParametersV1(m_fname_bkgd);
    else            m_bkgd = new ImgAlgos::ImgParametersV1(m_shape);   // zero array

    if( m_do_gain ) m_gain = new ImgAlgos::ImgParametersV1(m_fname_gain);
    else            m_gain = new ImgAlgos::ImgParametersV1(m_shape,1);// unit array

    m_peds_data = m_peds->data();
    m_bkgd_data = m_bkgd->data();
    m_gain_data = m_gain->data();
    m_mask_data = m_mask->data();

    if( m_print_bits & 4 ) m_peds -> print("Pedestals");
    if( m_print_bits & 4 ) m_mask -> print("Mask");
    if( m_print_bits & 4 ) m_bkgd -> print("Background");
    if( m_print_bits & 4 ) m_gain -> print("Gain");

    m_cdat = new double [m_size];
    std::fill_n(m_cdat, int(m_size), 0);    

    if( m_do_bkgd ) defImgIndexesForBkgdNorm();
}

//--------------------

void 
ImgCalib::procEvent(Event& evt, Env& env)
{
  shared_ptr< ndarray<double,2> > img = evt.get(m_str_src, m_key_in, &m_src);
  if (img.get()) {
    m_rdat = img->data();

    // Evaluate:
    // m_cdat[i] = (m_rdat[i] - m_peds[i] - m_norm*m_bkgd[i]) * m_gain[i]; and apply m_mask[i];

    memcpy(m_cdat,m_rdat,m_size*sizeof(double)); 
    if (m_do_peds) {             for(unsigned i=0; i<m_size; i++) m_cdat[i] -= m_peds_data[i]; }
    if (m_do_bkgd) { normBkgd(); for(unsigned i=0; i<m_size; i++) m_cdat[i] -= m_bkgd_data[i]*m_norm; }
    if (m_do_gain) {             for(unsigned i=0; i<m_size; i++) m_cdat[i] *= m_gain_data[i]; }
    if (m_do_mask) {             
      for(unsigned i=0; i<m_size; i++) {
        if (m_mask_data[i]==0) m_cdat[i] = m_mask_val; 
      }
    }
  } 
  else
  {
    const std::string msg = "Image is not available in the event(...) for source:" + m_str_src + " key:" + m_key_in;
    MsgLog(name(), info, msg);
  }
}

//--------------------

void
ImgCalib::normBkgd()
{
  double sum_data=0;
  double sum_bkgd=0;
  for(std::vector<unsigned>::const_iterator it = v_inds.begin(); it != v_inds.end(); ++ it) {
      sum_data += m_cdat[*it];
      sum_bkgd += m_bkgd->data()[*it];
  }
  m_norm = (sum_bkgd != 0)? (float)(sum_data/sum_bkgd) : 1;
}

//--------------------
 
void 
ImgCalib::defImgIndexesForBkgdNorm()
{
  v_inds.clear();
  for(unsigned r = m_row_min; r < m_row_max+1; r++) {
    for(unsigned c = m_col_min; c < m_col_max+1; c++) v_inds.push_back(r*m_cols+c);
  }
}

//--------------------

void 
ImgCalib::printEventRecord(Event& evt)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
  );
}

//--------------------
// This method defines the m_shape or throw message that can not do that.
void 
ImgCalib::defineImageShape(Event& evt)
{
  shared_ptr< ndarray<double,2> > img = evt.get(m_str_src, m_key_in, &m_src);
  if (img.get()) {
    //memcpy(m_shape,img->shape(),2*sizeof(unsigned));
    for(int i=0;i<2;i++) m_shape[i]=img->shape()[i];
    m_rows = m_shape[0];
    m_cols = m_shape[1];
    m_size = m_rows*m_cols;
    if( m_print_bits & 1 )
      MsgLog(name(), info, "Image shape (rows,cols) =" << m_shape[0] << ", " << m_shape[1]);
  } 
  else
  {
    const std::string msg = "Image shape is not defined in the event(...) for source:" + m_str_src + " key:" + m_key_in;
    MsgLogRoot(error, msg);
    throw std::runtime_error(msg);
  }
}

//--------------------

void 
ImgCalib::saveImageInEvent(Event& evt)
{
  shared_ptr< ndarray<double,2> > img2d( new ndarray<double,2>(m_cdat, m_shape) );
  evt.put(img2d, m_src, m_key_out);
}

//--------------------

//--------------------
} // namespace ImgAlgos
