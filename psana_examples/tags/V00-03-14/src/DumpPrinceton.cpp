//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpPrinceton...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpPrinceton.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/princeton.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpPrinceton)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpPrinceton::DumpPrinceton (const std::string& name)
  : Module(name)
{
  m_src = configSrc("source", "DetInfo(:Princeton)");
}

//--------------
// Destructor --
//--------------
DumpPrinceton::~DumpPrinceton ()
{
}

// Method which is called at the beginning of the calibration cycle
void 
DumpPrinceton::beginCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), trace, "in beginCalibCycle()");

  shared_ptr<Psana::Princeton::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {
    
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

  shared_ptr<Psana::Princeton::ConfigV2> config2 = env.configStore().get(m_src);
  if (config2) {
    
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

  shared_ptr<Psana::Princeton::ConfigV3> config3 = env.configStore().get(m_src);
  if (config3) {
    
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
      str << "\n  gainIndex = " << int(config3->gainIndex());
      str << "\n  readoutSpeedIndex = " << int(config3->readoutSpeedIndex());
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
}

// Method which is called with event data
void 
DumpPrinceton::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Princeton::FrameV1> frame1 = evt.get(m_src);
  if (frame1) {
    WithMsgLog(name(), info, str) {
      str << "Princeton::FrameV1:";
      str << "\n  shotIdStart = " << frame1->shotIdStart();
      str << "\n  readoutTime = " << frame1->readoutTime();
      str << "\n  data = " << frame1->data();
    }
  }

  shared_ptr<Psana::Princeton::FrameV2> frame2 = evt.get(m_src);
  if (frame2) {
    WithMsgLog(name(), info, str) {
      str << "Princeton::FrameV2:";
      str << "\n  shotIdStart = " << frame2->shotIdStart();
      str << "\n  readoutTime = " << frame2->readoutTime();
      str << "\n  temperature = " << frame2->temperature();
      str << "\n  data = " << frame2->data();
    }
  }

  shared_ptr<Psana::Princeton::InfoV1> info1 = evt.get(m_src);
  if (info1) {
    WithMsgLog(name(), info, str) {
      str << "Princeton::InfoV1:";
      str << "\n  temperature = " << info1->temperature();
    }
  }
}
  
} // namespace psana_examples
