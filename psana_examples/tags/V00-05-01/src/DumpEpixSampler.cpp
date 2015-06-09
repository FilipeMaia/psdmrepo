//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpEpixSampler...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpEpixSampler.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/epixsampler.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpEpixSampler)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpEpixSampler::DumpEpixSampler (const std::string& name)
  : Module(name)
  , m_src()
{
  m_src = configSrc("source", "DetInfo(:EpixSampler)");
}

//--------------
// Destructor --
//--------------
DumpEpixSampler::~DumpEpixSampler ()
{
}

/// Method which is called at the beginning of the run
void 
DumpEpixSampler::beginRun(Event& evt, Env& env)
{
  shared_ptr<Psana::EpixSampler::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {
    WithMsgLog(name(), info, str) {
      str << "Psana::EpixSampler::ConfigV1:";
      str << "\n  version = " << config1->version();
      str << "\n  runTrigDelay = " << config1->runTrigDelay();
      str << "\n  daqTrigDelay = " << config1->daqTrigDelay();
      str << "\n  daqSetting = " << config1->daqSetting();
      str << "\n  adcClkHalfT = " << config1->adcClkHalfT();
      str << "\n  adcPipelineDelay = " << config1->adcPipelineDelay();
      str << "\n  digitalCardId0 = " << config1->digitalCardId0();
      str << "\n  digitalCardId1 = " << config1->digitalCardId1();
      str << "\n  analogCardId0 = " << config1->analogCardId0();
      str << "\n  analogCardId1 = " << config1->analogCardId1();
      str << "\n  numberOfChannels = " << config1->numberOfChannels();
      str << "\n  samplesPerChannel = " << config1->samplesPerChannel();
      str << "\n  baseClockFrequency = " << config1->baseClockFrequency();
      str << "\n  testPatternEnable = " << int(config1->testPatternEnable());
    }
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpEpixSampler::event(Event& evt, Env& env)
{
  Pds::Src actualSrc;
  shared_ptr<Psana::EpixSampler::ElementV1> data1 = evt.get(m_src, "", &actualSrc);
  if (data1) {

    WithMsgLog(name(), info, str) {
      str << "EpixSampler::ElementV1 at " << actualSrc
          << "\n  vc = " << int(data1->vc())
          << "\n  lane = " << int(data1->lane())
          << "\n  acqCount = " << data1->acqCount()
          << "\n  frameNumber = " << data1->frameNumber()
          << "\n  ticks = " << data1->ticks()
          << "\n  fiducials = " << data1->fiducials()
          << "\n  temperatures = " << data1->temperatures()
          << "\n  frame = " << data1->frame();
    }
  }
}

} // namespace psana_examples
