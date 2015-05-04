//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraImageProducer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CameraImageProducer.h"

//-----------------
// C/C++ Headers --
//-----------------
//#include <iomanip> // for setw, setfill
//#include <sstream> // for stringstream
//#include <iostream>// for setf

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(CameraImageProducer)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace {
  
  void
  printFrameCoord(std::ostream& str, const Psana::Camera::FrameCoord& coord) 
  {
    str << "(" << coord.column() << ", " << coord.row() << ")";
  }
  
}

//--------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
CameraImageProducer::CameraImageProducer (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out() 
  , m_outtype()
  , m_subtract_offset()
  , m_print_bits()
  , m_count(0)
  , m_count_msg(0)
{
  // get the values from configuration or use defaults
  m_str_src           = configSrc("source", "DetInfo(:Camera)");
  m_key_in            = configStr("key_in",                 "");
  m_key_out           = configStr("key_out",           "image");
  m_outtype           = configStr("outtype",          "asdata");
  m_subtract_offset   = config   ("subtract_offset",      true);
  m_print_bits        = config   ("print_bits",             0 );

  m_detector = detectorTypeForSource(m_str_src);

  checkTypeImplementation();
  
}

//--------------------

void 
CameraImageProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters :"
        << "\n source           : " << m_str_src
        << "\n key_in           : " << m_key_in      
        << "\n key_out          : " << m_key_out
        << "\n outtype          : " << m_outtype
        << "\n subtract_offset  : " << m_subtract_offset     
        << "\n print_bits       : " << m_print_bits
        << "\n\n Derived enum parameters:"
        << "\n dtype            : " << m_dtype    
        << "\n detector         : " << m_detector    
        << "\n";     
  }
}

//--------------------

//--------------
// Destructor --
//--------------
CameraImageProducer::~CameraImageProducer ()
{
}

/// Method which is called once at the beginning of the job
void 
CameraImageProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

/// Method which is called at the beginning of the run
void CameraImageProducer::beginRun(Event& evt, Env& env) {}

/// Method which is called at the beginning of the calibration cycle
void 
CameraImageProducer::beginCalibCycle(Event& evt, Env& env) 
{
  if( m_print_bits & 16 ) {

      MsgLog(name(), info, "in beginCalibCycle()");
      
      shared_ptr<Psana::Camera::FrameFexConfigV1> config1 = env.configStore().get(m_src);
      if (config1) {
        WithMsgLog(name(), info, str) {
          str << "Camera::FrameFexConfigV1:";
          str << "\n  forwarding = " << config1->forwarding();
          str << "\n  forward_prescale = " << config1->forward_prescale();
          str << "\n  processing = " << config1->processing();
          str << "\n  roiBegin = ";
          ::printFrameCoord(str, config1->roiBegin());
          str << "\n  roiEnd = ";
          ::printFrameCoord(str, config1->roiEnd());
          str << "\n  threshold = " << config1->threshold();
          str << "\n  number_of_masked_pixels = " << config1->number_of_masked_pixels();
          const ndarray<const Psana::Camera::FrameCoord, 1>& masked_pixels = config1->masked_pixel_coordinates();
          for (unsigned i = 0; i < masked_pixels.shape()[0]; ++ i) {
            str << "\n    ";
            ::printFrameCoord(str, masked_pixels[i]);
          }
        }     
      
      } else {
        MsgLog(name(), info, "Camera::FrameFexConfigV1 not found");    
      }

  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CameraImageProducer::event(Event& evt, Env& env)
{
  ++ m_count;
  if( m_print_bits & 2 ) printEventRecord(evt);
  procEvent(evt,env);
}
  
/// Method which is called at the end of the calibration cycle
void CameraImageProducer::endCalibCycle(Event& evt, Env& env) {}

/// Method which is called at the end of the run
void CameraImageProducer::endRun(Event& evt, Env& env) {}

/// Method which is called once at the end of the job
void CameraImageProducer::endJob(Event& evt, Env& env)
{
  if( m_print_bits & 4 ) printSummary(evt);
}

//--------------------
//--------------------

void 
CameraImageProducer::procEvent(Event& evt, Env& env)
{
  // proc event  for one of the supported data types
  if ( m_dtype == ASDATA  and procEventForOutputType<data_t>  (evt) ) return; 
  if ( m_dtype == FLOAT   and procEventForOutputType<float>   (evt) ) return; 
  if ( m_dtype == DOUBLE  and procEventForOutputType<double>  (evt) ) return; 
  if ( m_dtype == INT     and procEventForOutputType<int>     (evt) ) return; 
  if ( m_dtype == INT16   and procEventForOutputType<int16_t> (evt) ) return; 
}

//--------------------

void 
CameraImageProducer::printEventRecord(Event& evt, std::string comment)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
                     << comment.c_str() 
  );
}

//--------------------

void 
CameraImageProducer::printSummary(Event& evt, std::string comment)
{
  MsgLog( name(), info, "Run=" << stringRunNumber(evt) 
	                << "Number of processed events=" << stringFromUint(m_count)
                        << comment.c_str() );
}

//--------------------

void 
CameraImageProducer::checkTypeImplementation()
{  
  if ( m_outtype == "asdata" ) { m_dtype = ASDATA; return; }
  if ( m_outtype == "float"  ) { m_dtype = FLOAT;  return; }
  if ( m_outtype == "double" ) { m_dtype = DOUBLE; return; } 
  if ( m_outtype == "int"    ) { m_dtype = INT;    return; } 
  if ( m_outtype == "int16"  ) { m_dtype = INT16;  return; } 

  const std::string msg = "The requested data type: " + m_outtype + " is not implemented";
  MsgLog(name(), warning, msg );
  throw std::runtime_error(msg);
}

//--------------------
} // namespace ImgAlgos
//--------------------
