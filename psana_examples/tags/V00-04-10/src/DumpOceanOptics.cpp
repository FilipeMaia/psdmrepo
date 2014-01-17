//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpOceanOptics...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpOceanOptics.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSTime/Time.h"
#include "psddl_psana/oceanoptics.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpOceanOptics)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpOceanOptics::DumpOceanOptics (const std::string& name)
  : Module(name)
  , m_src()
{
  // get the values from configuration or use defaults
  m_src = configSrc("source", "DetInfo(:OceanOptics)");
}

//--------------
// Destructor --
//--------------
DumpOceanOptics::~DumpOceanOptics ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpOceanOptics::beginCalibCycle(Event& evt, Env& env)
{
  shared_ptr<Psana::OceanOptics::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {

    WithMsgLog(name(), info, str) {
      str << "Psana::OceanOptics::ConfigV1:";
      str << "\n  exposureTime = " << config1->exposureTime();
      str << "\n  strayLightConstant = " << config1->strayLightConstant();
      str << "\n  waveLenCalib = " << config1->waveLenCalib();
      str << "\n  nonlinCorrect = " << config1->nonlinCorrect();
    }

  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpOceanOptics::event(Event& evt, Env& env)
{
  shared_ptr<Psana::OceanOptics::DataV1> data1 = evt.get(m_src);
  if (data1) {

    WithMsgLog(name(), info, str) {
      str << "Psana::OceanOptics::DataV1:";
      str << "\n  data = " << data1->data();
      str << "\n  frameCounter = " << data1->frameCounter();
      str << "\n  numDelayedFrames = " << data1->numDelayedFrames();
      str << "\n  numDiscardFrames = " << data1->numDiscardFrames();
      str << "\n  timeFrameStart =     " << PSTime::Time(data1->timeFrameStart().tv_sec(), data1->timeFrameStart().tv_nsec());
      str << "\n  timeFrameFirstData = " << PSTime::Time(data1->timeFrameFirstData().tv_sec(), data1->timeFrameFirstData().tv_nsec());
      str << "\n  timeFrameEnd =       " << PSTime::Time(data1->timeFrameEnd().tv_sec(), data1->timeFrameEnd().tv_nsec());
      str << "\n  numSpectraInData = " << int(data1->numSpectraInData());
      str << "\n  numSpectraInQueue = " << int(data1->numSpectraInQueue());
      str << "\n  numSpectraUnused = " << int(data1->numSpectraUnused());
      str << "\n  durationOfFrame = " << data1->durationOfFrame();
    }

  }
}

} // namespace psana_examples
