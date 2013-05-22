//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpImp...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpImp.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/imp.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpImp)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpImp::DumpImp (const std::string& name)
  : Module(name)
  , m_src()
{
  m_src = configSrc("source", "DetInfo(:Imp)");
}

//--------------
// Destructor --
//--------------
DumpImp::~DumpImp ()
{
}

/// Method which is called at the beginning of the run
void 
DumpImp::beginRun(Event& evt, Env& env)
{
  MsgLog(name(), info, "in beginRun()");

  Pds::Src actualSrc;
  shared_ptr<Psana::Imp::ConfigV1> config1 = env.configStore().get(m_src, &actualSrc);
  if (config1) {
    WithMsgLog(name(), info, str) {
      str << "Imp::ConfigV1 at " << actualSrc;
      str << "\n  range = " << config1->range();
      str << "\n  calRange = " << config1->calRange();
      str << "\n  reset = " << config1->reset();
      str << "\n  biasData = " << config1->biasData();
      str << "\n  calData = " << config1->calData();
      str << "\n  biasDacData = " << config1->biasDacData();
      str << "\n  calStrobe = " << config1->calStrobe();
      str << "\n  numberOfSamples = " << config1->numberOfSamples();
      str << "\n  trigDelay = " << config1->trigDelay();
      str << "\n  adcDelay = " << config1->adcDelay();
    }
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpImp::event(Event& evt, Env& env)
{
  Pds::Src actualSrc;
  shared_ptr<Psana::Imp::ElementV1> data1 = evt.get(m_src, "", &actualSrc);
  if (data1) {

    WithMsgLog(name(), info, str) {
      str << "Imp::ElementV1 at " << actualSrc
          << "\n  vc = " << int(data1->vc())
          << "\n  lane = " << int(data1->lane())
          << "\n  frameNumber = " << data1->frameNumber()
          << "\n  range = " << data1->range();

      Psana::Imp::LaneStatus laneStatus = data1->laneStatus();
      str << "\n  laneStatus.linkErrCount = " << int(laneStatus.linkErrCount())
          << "\n  laneStatus.linkDownCount = " << int(laneStatus.linkDownCount())
          << "\n  laneStatus.cellErrCount = " << int(laneStatus.cellErrCount())
          << "\n  laneStatus.rxCount = " << int(laneStatus.rxCount())
          << "\n  laneStatus.locLinked = " << int(laneStatus.locLinked())
          << "\n  laneStatus.remLinked = " << int(laneStatus.remLinked())
          << "\n  laneStatus.zeros = " << int(laneStatus.zeros())
          << "\n  laneStatus.powersOkay = " << int(laneStatus.powersOkay());

      // get samples 1-d array
      const ndarray<const Psana::Imp::Sample, 1>& samples = data1->samples();
      for (unsigned i = 0; i != samples.size(); ++ i) {
        str << "\n  sample[" << i << "]: channels = " << samples[i].channels();
      }

    }
  }
}
  
} // namespace psana_examples
