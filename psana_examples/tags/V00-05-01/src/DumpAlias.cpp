//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpAlias...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpAlias.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/alias.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpAlias)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpAlias::DumpAlias (const std::string& name)
  : Module(name)
  , m_src()
{
  // get the values from configuration or use defaults
  m_src = configSrc("source", "ProcInfo()");
}

//--------------
// Destructor --
//--------------
DumpAlias::~DumpAlias ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpAlias::beginCalibCycle(Event& evt, Env& env)
{
  shared_ptr<Psana::Alias::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {

    WithMsgLog(name(), info, str) {
      str << "Alias::ConfigV1:";
      str << "\n  numSrcAlias = " << config1->numSrcAlias();
      const ndarray<const Psana::Alias::SrcAlias, 1>& srcAlias = config1->srcAlias();
      for (unsigned i = 0; i != srcAlias.size(); ++ i) {
        const Psana::Alias::SrcAlias& alias = srcAlias[i];
        str << "\n    " << i << ": " << alias.aliasName() << " -> " << alias.src() ;
      }
    }

  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpAlias::event(Event& evt, Env& env)
{
}
  
} // namespace psana_examples
