//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpAcqiris...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpAcqiris.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/DetInfo.hh"
#include "psddl_psana/acqiris.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpAcqiris)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpAcqiris::DumpAcqiris (const std::string& name)
  : Module(name)
{
  // get the values from configuration or use defaults
  m_src = configStr("source", "DetInfo(:Acqiris)");
}

//--------------
// Destructor --
//--------------
DumpAcqiris::~DumpAcqiris ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpAcqiris::beginCalibCycle(Env& env)
{
  MsgLog(name(), trace, "in beginCalibCycle()");

  shared_ptr<Psana::Acqiris::ConfigV1> acqConfig = env.configStore().get(m_src);
  if (acqConfig.get()) {
    MsgLog(name(), info, "Acqiris::ConfigV1: nbrBanks=" << acqConfig->nbrBanks()
           << " channelMask=" << acqConfig->channelMask()
           << " nbrChannels=" << acqConfig->nbrChannels()
           << " h.sampInterval=" << acqConfig->horiz().sampInterval()
           << " h.delayTime=" << acqConfig->horiz().delayTime()
           << " h.nbrSamples=" << acqConfig->horiz().nbrSamples()
           << " h.nbrSegments=" << acqConfig->horiz().nbrSegments()
           << " v[0].fullScale=" << acqConfig->vert(0).fullScale()
           << " v[0].offset=" << acqConfig->vert(0).offset()
           << " v[1].fullScale=" << acqConfig->vert(1).fullScale()
           << " v[1].offset=" << acqConfig->vert(1).offset()
           );
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpAcqiris::event(Event& evt, Env& env)
{

  shared_ptr<Psana::Acqiris::DataDescV1> acqData = evt.get(m_src);
  if (acqData.get()) {
    const std::vector<int>& shape = acqData->data_shape();
    for (int i = 0; i < shape[0]; ++ i) {
      const Psana::Acqiris::DataDescV1Elem& elem = acqData->data(i);
      MsgLog(name(), info, "Acqiris::DataDescV1: element=" << i 
           << " nbrSegments=" << elem.nbrSegments()
           << " nbrSamplesInSeg= " << elem.nbrSamplesInSeg());
    }
  }
}

} // namespace psana_examples
