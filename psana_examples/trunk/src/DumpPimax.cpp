//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DumpPimax.cpp 5929 2014-03-17 09:25:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class DumpPimax...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpPimax.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/pimax.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpPimax)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpPimax::DumpPimax (const std::string& name)
  : Module(name)
{
  m_src = configSrc("source", "DetInfo(:Pimax)");
}

//--------------
// Destructor --
//--------------
DumpPimax::~DumpPimax ()
{
}

// Method which is called at the beginning of the calibration cycle
void 
DumpPimax::beginCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), trace, "in beginCalibCycle()");

  shared_ptr<Psana::Pimax::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {
    
    WithMsgLog(name(), info, str) {
      str << "Pimax::ConfigV1:";
      str << "\n  width = " << config1->width();
      str << "\n  height = " << config1->height();
      str << "\n  orgX = " << config1->orgX();
      str << "\n  orgY = " << config1->orgY();
      str << "\n  binX = " << config1->binX();
      str << "\n  binY = " << config1->binY();
      str << "\n  exposureTime = " << config1->exposureTime();
      str << "\n  coolingTemp = " << config1->coolingTemp();
      str << "\n  readoutSpeed = " << config1->readoutSpeed();
      str << "\n  gainIndex = " << config1->gainIndex();
      str << "\n  intensifierGain = " << config1->intensifierGain();
      str << "\n  gateDelay = " << config1->gateDelay();
      str << "\n  gateWidth = " << config1->gateWidth();
      str << "\n  maskedHeight = " << config1->maskedHeight();
      str << "\n  kineticHeight = " << config1->kineticHeight();
      str << "\n  vsSpeed = " << config1->vsSpeed();
      str << "\n  infoReportInterval = " << config1->infoReportInterval();
      str << "\n  exposureEventCode = " << config1->exposureEventCode();
      str << "\n  numIntegrationShots = " << config1->numIntegrationShots();
      str << "\n  frameSize = " << config1->frameSize();
      str << "\n  numPixelsX = " << config1->numPixelsX();
      str << "\n  numPixelsY = " << config1->numPixelsY();
      str << "\n  numPixels = " << config1->numPixels();
    }
    
  }
}

// Method which is called with event data
void 
DumpPimax::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Pimax::FrameV1> frame1 = evt.get(m_src);
  if (frame1) {
    WithMsgLog(name(), info, str) {
      str << "Pimax::FrameV1:";
      str << "\n  shotIdStart = " << frame1->shotIdStart();
      str << "\n  readoutTime = " << frame1->readoutTime();
      str << "\n  temperature = " << frame1->temperature();
      str << "\n  data = " << frame1->data();
    }
  }
}
  
} // namespace psana_examples
