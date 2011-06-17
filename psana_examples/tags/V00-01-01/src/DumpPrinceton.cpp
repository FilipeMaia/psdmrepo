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
  m_src = configStr("source", "DetInfo(:Princeton)");
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
}

// Method which is called with event data
void 
DumpPrinceton::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Princeton::FrameV1> frame = evt.get(m_src);
  if (frame.get()) {
    WithMsgLog(name(), info, str) {
      str << "Princeton::FrameV1:";
      str << "\n  readoutTime = " << frame->readoutTime();

      const uint16_t* data = frame->data();
      str << "\n  data =";
      for (int i = 0; i < 10; ++ i) {
        str << " " << data[i];
      }
      str << " ...";
    }
  }
}
  
} // namespace psana_examples
