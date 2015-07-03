//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpRayonix...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpRayonix.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/rayonix.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpRayonix)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpRayonix::DumpRayonix (const std::string& name)
  : Module(name)
  , m_src()
{
  // get the values from configuration or use defaults
  m_src = configSrc("source", "DetInfo(:Rayonix)");
}

//--------------
// Destructor --
//--------------
DumpRayonix::~DumpRayonix ()
{
}

/// Method which is called at the beginning of the run
void 
DumpRayonix::beginRun(Event& evt, Env& env)
{
  shared_ptr<Psana::Rayonix::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {
    WithMsgLog(name(), info, str) {
      str << "Psana::Rayonix::ConfigV1:";
      str << "\n  binning_f = " << int(config1->binning_f());
      str << "\n  binning_s = " << int(config1->binning_s());
      str << "\n  exposure = " << config1->exposure();
      str << "\n  trigger = " << config1->trigger();
      str << "\n  rawMode = " << config1->rawMode();
      str << "\n  darkFlag = " << config1->darkFlag();
      str << "\n  readoutMode = " << config1->readoutMode();
      str << "\n  deviceID = " << config1->deviceID();
    }
  }

  shared_ptr<Psana::Rayonix::ConfigV2> config2 = env.configStore().get(m_src);
  if (config2) {
    WithMsgLog(name(), info, str) {
      str << "Psana::Rayonix::ConfigV2:";
      str << "\n  binning_f = " << int(config2->binning_f());
      str << "\n  binning_s = " << int(config2->binning_s());
      str << "\n  testPattern = " << config2->testPattern();
      str << "\n  exposure = " << config2->exposure();
      str << "\n  trigger = " << config2->trigger();
      str << "\n  rawMode = " << config2->rawMode();
      str << "\n  darkFlag = " << config2->darkFlag();
      str << "\n  readoutMode = " << config2->readoutMode();
      str << "\n  deviceID = " << config2->deviceID();
    }
  }

}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpRayonix::event(Event& evt, Env& env)
{
}
  
} // namespace psana_examples
