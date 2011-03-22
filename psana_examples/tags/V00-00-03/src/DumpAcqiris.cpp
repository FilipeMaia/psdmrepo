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

namespace {
  
  // name of the logger to be used with MsgLogger
  const char* logger = "DumpAcqiris"; 
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpAcqiris::DumpAcqiris (const std::string& name)
  : Module(name)
  , m_maxEvents()
  , m_filter()
{
  // get the values from configuration or use defaults
  m_maxEvents = config("events", 32U);
  m_filter = config("filter", false);
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
  MsgLog(logger, info, name() << ": in beginCalibCycle()");

  // Get Psana::Acqiris::ConfigV1 data
  Pds::DetInfo address(0, Pds::DetInfo::AmoGasdet, 0, Pds::DetInfo::Acqiris, 0);

  shared_ptr<Psana::Acqiris::ConfigV1> acqConfig = env.configStore().get(address);
  if (not acqConfig.get()) {
    MsgLog(logger, info, name() << ": Acqiris::ConfigV1 not found");    
  } else {
    MsgLog(logger, info, name() << ": Acqiris::ConfigV1: nbrBanks=" << acqConfig->nbrBanks()
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
  Pds::DetInfo address(0, Pds::DetInfo::AmoGasdet, 0, Pds::DetInfo::Acqiris, 0);
  
  shared_ptr<Psana::Acqiris::DataDescV1> acqData = evt.get(address);
  if (not acqData.get()) {
    MsgLog(logger, info, name() << ": Acqiris::DataDescV1 not found");    
  } else {
    const std::vector<int>& shape = acqData->_data_shape();
    for (int i = 0; i < shape[0]; ++ i) {
      const Psana::Acqiris::DataDescV1Elem& elem = acqData->data(i);
      MsgLog(logger, info, name() << ": Acqiris::DataDescV1: element=" << i 
           << " nbrSegments=" << elem.nbrSegments()
           << " nbrSamplesInSeg= " << elem.nbrSamplesInSeg());
    }
  }
}

} // namespace psana_examples
