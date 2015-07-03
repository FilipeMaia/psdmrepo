//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpOrca...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpOrca.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/orca.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpOrca)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpOrca::DumpOrca (const std::string& name)
  : Module(name)
  , m_src()
{
  // get the values from configuration or use defaults
  m_src = configSrc("source", "DetInfo(:OrcaFl40)");
}

//--------------
// Destructor --
//--------------
DumpOrca::~DumpOrca ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpOrca::beginCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), trace, "in beginCalibCycle()");

  shared_ptr<Psana::Orca::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {

    WithMsgLog(name(), info, str) {
      str << "Orca::ConfigV1:";
      str << "\n  mode = " << int(config1->mode());
      str << "\n  rows = " << config1->rows();
      str << "\n  cooling = " << int(config1->cooling());
      str << "\n  defect_pixel_correction_enabled = " << int(config1->defect_pixel_correction_enabled());
    }

  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpOrca::event(Event& evt, Env& env)
{
}

} // namespace psana_examples
