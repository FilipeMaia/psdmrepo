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
  , m_fname_mask()
  , m_fname_bkgd()
  , m_fname_gain()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_str_src           = configStr("source", "DetInfo(:Camera)");
  m_key_in            = configStr("key_in",                 "");
  m_key_out           = configStr("key_out",  "calibrated_img");
  m_fname_peds        = configStr("fname_peds",             "");
  m_fname_mask        = configStr("fname_mask",             "");
  m_fname_bkgd        = configStr("fname_bkgd",             "");
  m_fname_gain        = configStr("fname_gain",             "");
  m_print_bits        = config   ("print_bits",             0 );
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

    if( m_fname_peds.empty() ) m_peds = new ImgAlgos::ImgParametersV1(m_shape);   // zero array
    else                       m_peds = new ImgAlgos::ImgParametersV1(m_fname_peds);

    if( m_fname_mask.empty() ) m_mask = new ImgAlgos::ImgParametersV1(m_shape,1); // unit array
    else                       m_mask = new ImgAlgos::ImgParametersV1(m_fname_mask);

    if( m_fname_bkgd.empty() ) m_bkgd = new ImgAlgos::ImgParametersV1(m_shape);   // zero array
    else                       m_bkgd = new ImgAlgos::ImgParametersV1(m_fname_bkgd);

    if( m_fname_gain.empty() ) m_gain = new ImgAlgos::ImgParametersV1(m_shape,1); // unit array
    else                       m_gain = new ImgAlgos::ImgParametersV1(m_fname_gain);

    if( m_print_bits & 4 ) m_peds -> print("Pedestals");
    if( m_print_bits & 4 ) m_mask -> print("Mask");
    if( m_print_bits & 4 ) m_bkgd -> print("Background");
    if( m_print_bits & 4 ) m_gain -> print("Gain");
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
    memcpy(m_shape,img->shape(),2*sizeof(unsigned));
    m_size = m_shape[0]*m_shape[1]; 
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
ImgCalib::saveImageInEvent(Event& evt, double *p_data, const unsigned *shape)
{
  shared_ptr< ndarray<double,2> > img2d( new ndarray<double,2>(p_data, m_shape) );
  evt.put(img2d, m_src, m_key_out);
}

//--------------------

//--------------------
} // namespace ImgAlgos
