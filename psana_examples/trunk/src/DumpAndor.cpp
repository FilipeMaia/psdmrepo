//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpAndor...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpAndor.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/andor.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpAndor)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpAndor::DumpAndor (const std::string& name)
  : Module(name)
  , m_src()
{
  m_src = configSrc("source", "DetInfo(:Andor)");
}

//--------------
// Destructor --
//--------------
DumpAndor::~DumpAndor ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpAndor::beginCalibCycle(Event& evt, Env& env)
{
  shared_ptr<Psana::Andor::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {

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

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpAndor::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Andor::FrameV1> frame1 = evt.get(m_src);
  if (frame1) {
    WithMsgLog(name(), info, str) {
      str << "Andor::FrameV1:";
      str << "\n  shotIdStart = " << frame1->shotIdStart();
      str << "\n  readoutTime = " << frame1->readoutTime();
      str << "\n  temperature = " << frame1->temperature();
      str << "\n  data = " << frame1->data();
    }
  }
}

} // namespace psana_examples
