//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: AndorImageProducer.cpp 0001 2014-01-17 09:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class AndorImageProducer...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/AndorImageProducer.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(AndorImageProducer)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
AndorImageProducer::AndorImageProducer (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out()
  , m_outtype()
  , m_print_bits()
  , m_count(0)
  , m_count_msg(0)
{
    m_str_src    = configSrc("source",     "DetInfo(:Andor)");  // DetInfo(MecTargetChamber.0:Andor.1)
    m_key_in     = configStr("key_in",     "");
    m_key_out    = configStr("key_out",    "andorimg");
    m_outtype    = configStr("outtype",    "asdata");
    m_print_bits = config   ("print_bits", 0);

    checkTypeImplementation();
}

//--------------
// Destructor --
//--------------
AndorImageProducer::~AndorImageProducer ()
{
}

//--------------------

/// Method which is called once at the beginning of the job
void 
AndorImageProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
  if( m_print_bits & 8 ) printSizeOfTypes();
}

// Method which is called at the beginning of the calibration cycle
void 
AndorImageProducer::beginCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), trace, "in beginCalibCycle()");

  shared_ptr<Psana::Andor::ConfigV1> config1 = env.configStore().get(m_str_src);
  if (config1) {

    if( m_print_bits & 4 ) {

      WithMsgLog(name(), info, str) {
        str << "Andor::ConfigV1:";
        str << "\n  width = " << config1->width();
        str << "\n  height = " << config1->height();
        str << "\n  orgX = " << config1->orgX();
        str << "\n  orgY = " << config1->orgY();
        str << "\n  binX = " << config1->binX();
        str << "\n  binY = " << config1->binY();
        str << "\n  exposureTime = " << config1->exposureTime();
        str << "\n  coolingTemp = " << config1->coolingTemp();
        str << "\n  fanMode = " << int(config1->fanMode());
        str << "\n  baselineClamp = " << int(config1->baselineClamp());
        str << "\n  highCapacity = " << int(config1->highCapacity());
        str << "\n  gainIndex = " << int(config1->gainIndex());
        str << "\n  readoutSpeedIndex = " << config1->readoutSpeedIndex();
        str << "\n  exposureEventCode = " << config1->exposureEventCode();
        str << "\n  numDelayShots = " << config1->numDelayShots();
        str << "\n  frameSize = " << config1->frameSize();
        str << "\n  numPixels = " << config1->numPixels();
      }
    }
  }
}

//--------------------
// Method which is called with event data
void 
AndorImageProducer::event(Event& evt, Env& env)
{
  ++ m_count;
  procEvent(evt, env);
}

//--------------------

void 
AndorImageProducer::procEvent(Event& evt, Env& env)
{  
  // proc event  for one of the supported data types
  if ( m_dtype == ASDATA  and procEventForOutputType<data_t> (evt) ) return; 
  if ( m_dtype == FLOAT   and procEventForOutputType<float>  (evt) ) return; 
  if ( m_dtype == DOUBLE  and procEventForOutputType<double> (evt) ) return; 
  if ( m_dtype == INT     and procEventForOutputType<int>    (evt) ) return; 
  if ( m_dtype == INT16   and procEventForOutputType<int16_t>(evt) ) return;
}

//--------------------

void 
AndorImageProducer::checkTypeImplementation()
{  
  if ( m_outtype == "asdata"  ) { m_dtype = ASDATA; return; }
  if ( m_outtype == "float"   ) { m_dtype = FLOAT;  return; }
  if ( m_outtype == "double"  ) { m_dtype = DOUBLE; return; } 
  if ( m_outtype == "int"     ) { m_dtype = INT;    return; } 
  if ( m_outtype == "int16"   ) { m_dtype = INT16;  return; } 

  const std::string msg = "The requested data type: " + m_outtype + " is not implemented";
  MsgLog(name(), warning, msg );
  throw std::runtime_error(msg);
}

//--------------------
/// Print input parameters
void 
AndorImageProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source     : "     << m_str_src
        << "\n key_in     : "     << m_key_in      
        << "\n key_out    : "     << m_key_out
        << "\n outtype    : "     << m_outtype
        << "\n print_bits : "     << m_print_bits
        << "\n";
  }
}

//--------------------

} // namespace ImgAlgos
