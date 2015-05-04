//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonImageProducer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/PrincetonImageProducer.h"

//-----------------
// C/C++ Headers --
//-----------------
//#include <iomanip> // for setw, setfill
//#include <sstream> // for stringstream
//#include <iostream>// for setf

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(PrincetonImageProducer)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
PrincetonImageProducer::PrincetonImageProducer (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out() 
  , m_outtype()
  , m_print_bits()
  , m_count(0)
  , m_count_msg(0)
{
  // get the values from configuration or use defaults
  m_str_src           = configSrc("source", "DetInfo(:Princeton)");
  m_key_in            = configStr("key_in",                    "");
  m_key_out           = configStr("key_out",              "image");
  m_outtype           = configStr("outtype",             "asdata");
  m_print_bits        = config   ("print_bits",                0 );

  checkTypeImplementation();
}

//--------------------

void 
PrincetonImageProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters :"
        << "\n source           : " << m_str_src
        << "\n key_in           : " << m_key_in      
        << "\n key_out          : " << m_key_out
        << "\n outtype          : " << m_outtype
        << "\n print_bits       : " << m_print_bits
        << "\n";     
  }
}

//--------------------

//--------------
// Destructor --
//--------------
PrincetonImageProducer::~PrincetonImageProducer ()
{
}

/// Method which is called once at the beginning of the job
void 
PrincetonImageProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

/// Method which is called at the beginning of the run
void 
PrincetonImageProducer::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
PrincetonImageProducer::beginCalibCycle(Event& evt, Env& env)
{
  if( m_print_bits & 16 ) {

    MsgLog(name(), info, "in beginCalibCycle()");
    
    shared_ptr<Psana::Princeton::ConfigV1> config1 = env.configStore().get(m_str_src);
    if (config1.get()) {    
      WithMsgLog(name(), info, str) {
        str << "Princeton::ConfigV1:";
        str << "\n  width = " << config1->width();
        str << "\n  height = " << config1->height();
        str << "\n  orgX = " << config1->orgX();
        str << "\n  orgY = " << config1->orgY();
        str << "\n  binX = " << config1->binX();
        str << "\n  binY = " << config1->binY();
        str << "\n  exposureTime = " << config1->exposureTime();
        str << "\n  coolingTemp = " << config1->coolingTemp();
        str << "\n  readoutSpeedIndex = " << config1->readoutSpeedIndex();
        str << "\n  readoutEventCode = " << config1->readoutEventCode();
        str << "\n  delayMode = " << config1->delayMode();
        str << "\n  frameSize = " << config1->frameSize();
        str << "\n  numPixels = " << config1->numPixels();
      }
    }
    
    shared_ptr<Psana::Princeton::ConfigV2> config2 = env.configStore().get(m_str_src);
    if (config2.get()) {
      WithMsgLog(name(), info, str) {
        str << "Princeton::ConfigV2:";
        str << "\n  width = " << config2->width();
        str << "\n  height = " << config2->height();
        str << "\n  orgX = " << config2->orgX();
        str << "\n  orgY = " << config2->orgY();
        str << "\n  binX = " << config2->binX();
        str << "\n  binY = " << config2->binY();
        str << "\n  exposureTime = " << config2->exposureTime();
        str << "\n  coolingTemp = " << config2->coolingTemp();
        str << "\n  gainIndex = " << config2->gainIndex();
        str << "\n  readoutSpeedIndex = " << config2->readoutSpeedIndex();
        str << "\n  readoutEventCode = " << config2->readoutEventCode();
        str << "\n  delayMode = " << config2->delayMode();
        str << "\n  frameSize = " << config2->frameSize();
        str << "\n  numPixels = " << config2->numPixels();
      }    
    }
    
    shared_ptr<Psana::Princeton::ConfigV3> config3 = env.configStore().get(m_str_src);
    if (config3.get()) {    
      WithMsgLog(name(), info, str) {
        str << "Princeton::ConfigV3:";
        str << "\n  width = " << config3->width();
        str << "\n  height = " << config3->height();
        str << "\n  orgX = " << config3->orgX();
        str << "\n  orgY = " << config3->orgY();
        str << "\n  binX = " << config3->binX();
        str << "\n  binY = " << config3->binY();
        str << "\n  exposureTime = " << config3->exposureTime();
        str << "\n  coolingTemp = " << config3->coolingTemp();
        str << "\n  gainIndex = " << config3->gainIndex();
        str << "\n  readoutSpeedIndex = " << config3->readoutSpeedIndex();
        str << "\n  exposureEventCode = " << config3->exposureEventCode();
        str << "\n  numDelayShots = " << config3->numDelayShots();
        str << "\n  frameSize = " << config3->frameSize();
        str << "\n  numPixels = " << config3->numPixels();
      } 
    }

    shared_ptr<Psana::Princeton::ConfigV4> config4 = env.configStore().get(m_src);
    if (config4) {
      WithMsgLog(name(), info, str) {
        str << "Princeton::ConfigV4:";
        str << "\n  width = " << config4->width();
        str << "\n  height = " << config4->height();
        str << "\n  orgX = " << config4->orgX();
        str << "\n  orgY = " << config4->orgY();
        str << "\n  binX = " << config4->binX();
        str << "\n  binY = " << config4->binY();
        str << "\n  maskedHeight = " << config4->maskedHeight();
        str << "\n  kineticHeight = " << config4->kineticHeight();
        str << "\n  vsSpeed = " << config4->vsSpeed();
        str << "\n  exposureTime = " << config4->exposureTime();
        str << "\n  coolingTemp = " << config4->coolingTemp();
        str << "\n  gainIndex = " << int(config4->gainIndex());
        str << "\n  readoutSpeedIndex = " << int(config4->readoutSpeedIndex());
        str << "\n  exposureEventCode = " << config4->exposureEventCode();
        str << "\n  numDelayShots = " << config4->numDelayShots();
        str << "\n  frameSize = " << config4->frameSize();
        str << "\n  numPixels = " << config4->numPixels();
      }
    }
  
    shared_ptr<Psana::Princeton::ConfigV5> config5 = env.configStore().get(m_src);
    if (config5) {
      WithMsgLog(name(), info, str) {
        str << "Princeton::ConfigV5:";
        str << "\n  width = " << config5->width();
        str << "\n  height = " << config5->height();
        str << "\n  orgX = " << config5->orgX();
        str << "\n  orgY = " << config5->orgY();
        str << "\n  binX = " << config5->binX();
        str << "\n  binY = " << config5->binY();
        str << "\n  exposureTime = " << config5->exposureTime();
        str << "\n  coolingTemp = " << config5->coolingTemp();
        str << "\n  gainIndex = " << config5->gainIndex();
        str << "\n  readoutSpeedIndex = " << config5->readoutSpeedIndex();
        str << "\n  maskedHeight = " << config5->maskedHeight();
        str << "\n  kineticHeight = " << config5->kineticHeight();
        str << "\n  vsSpeed = " << config5->vsSpeed();
        str << "\n  infoReportInterval = " << config5->infoReportInterval();
        str << "\n  exposureEventCode = " << config5->exposureEventCode();
        str << "\n  numDelayShots = " << config5->numDelayShots();
        str << "\n  frameSize = " << config5->frameSize();
        str << "\n  numPixels = " << config5->numPixels();
      }
    }

    shared_ptr<Psana::Pimax::ConfigV1> config1_pimax = env.configStore().get(m_src);
    if (config1_pimax) {    
      WithMsgLog(name(), info, str) {
        str << "Pimax::ConfigV1:";
        str << "\n  width = " << config1_pimax->width();
        str << "\n  height = " << config1_pimax->height();
        str << "\n  orgX = " << config1_pimax->orgX();
        str << "\n  orgY = " << config1_pimax->orgY();
        str << "\n  binX = " << config1_pimax->binX();
        str << "\n  binY = " << config1_pimax->binY();
        str << "\n  exposureTime = " << config1_pimax->exposureTime();
        str << "\n  coolingTemp = " << config1_pimax->coolingTemp();
        str << "\n  readoutSpeed = " << config1_pimax->readoutSpeed();
        str << "\n  gainIndex = " << config1_pimax->gainIndex();
        str << "\n  intensifierGain = " << config1_pimax->intensifierGain();
        str << "\n  gateDelay = " << config1_pimax->gateDelay();
        str << "\n  gateWidth = " << config1_pimax->gateWidth();
        str << "\n  maskedHeight = " << config1_pimax->maskedHeight();
        str << "\n  kineticHeight = " << config1_pimax->kineticHeight();
        str << "\n  vsSpeed = " << config1_pimax->vsSpeed();
        str << "\n  infoReportInterval = " << config1_pimax->infoReportInterval();
        str << "\n  exposureEventCode = " << config1_pimax->exposureEventCode();
        str << "\n  numIntegrationShots = " << config1_pimax->numIntegrationShots();
        str << "\n  frameSize = " << config1_pimax->frameSize();
        str << "\n  numPixelsX = " << config1_pimax->numPixelsX();
        str << "\n  numPixelsY = " << config1_pimax->numPixelsY();
        str << "\n  numPixels = " << config1_pimax->numPixels();
      }    
    }

  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
PrincetonImageProducer::event(Event& evt, Env& env)
{
  ++ m_count;
  if( m_print_bits & 2 ) printEventRecord(evt);
  procEvent(evt,env);
}
  
/// Method which is called at the end of the calibration cycle
void 
PrincetonImageProducer::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
PrincetonImageProducer::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
PrincetonImageProducer::endJob(Event& evt, Env& env)
{
  if( m_print_bits & 4 ) printSummary(evt);
}

//--------------------
//--------------------

void 
PrincetonImageProducer::procEvent(Event& evt, Env& env)
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
PrincetonImageProducer::printEventRecord(Event& evt, std::string comment)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Time="   << stringTimeStamp(evt) 
	             << comment.c_str() 
  );
}

//--------------------

void 
PrincetonImageProducer::printSummary(Event& evt, std::string comment)
{
  MsgLog( name(), info, "Run=" << stringRunNumber(evt) 
	                << " Number of processed events=" << stringFromUint(m_count)
                        << comment.c_str()
  );
}

//--------------------

void 
PrincetonImageProducer::checkTypeImplementation()
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
