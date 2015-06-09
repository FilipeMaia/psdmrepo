//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpL3T...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpL3T.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/l3t.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpL3T)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpL3T::DumpL3T (const std::string& name)
  : Module(name)
  , m_src()
{
  m_src = configSrc("source", "ProcInfo()");
}

//--------------
// Destructor --
//--------------
DumpL3T::~DumpL3T ()
{
}

/// Method which is called at the beginning of the run
void 
DumpL3T::beginRun(Event& evt, Env& env)
{
  shared_ptr<Psana::L3T::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {
    WithMsgLog(name(), info, str) {
      str << "L3T::ConfigV1:";
      str << "\n  module_id = \"" << config1->module_id() << "\"";
      str << "\n  description = \"" << config1->desc() << "\"";
    }
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpL3T::event(Event& evt, Env& env)
{
  // example of getting detector data from event
  shared_ptr<Psana::L3T::DataV1> data1 = evt.get(m_src);
  if (data1) {
    MsgLog(name(), info, "L3T::DataV1: accept = " << int(data1->accept()));
  }
}
  
} // namespace psana_examples
