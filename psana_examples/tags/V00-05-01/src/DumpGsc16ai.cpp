//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpGsc16ai...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpGsc16ai.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/gsc16ai.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpGsc16ai)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpGsc16ai::DumpGsc16ai (const std::string& name)
  : Module(name)
  , m_src()
{
  // get the values from configuration or use defaults
  m_src = configSrc("source", "DetInfo(:Gsc16ai)");
}

//--------------
// Destructor --
//--------------
DumpGsc16ai::~DumpGsc16ai ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpGsc16ai::beginCalibCycle(Event& evt, Env& env)
{
  shared_ptr<Psana::Gsc16ai::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {

    WithMsgLog(name(), info, str) {
      str << "Psana::Gsc16ai::ConfigV1:";
      str << "\n  voltageRange = " << config1->voltageRange();
      str << "\n  firstChan = " << config1->firstChan();
      str << "\n  lastChan = " << config1->lastChan();
      str << "\n  inputMode = " << config1->inputMode();
      str << "\n  triggerMode = " << config1->triggerMode();
      str << "\n  dataFormat = " << config1->dataFormat();
      str << "\n  fps = " << config1->fps();
      str << "\n  autocalibEnable = " << int(config1->autocalibEnable());
      str << "\n  timeTagEnable = " << int(config1->timeTagEnable());
      str << "\n  numChannels = " << config1->numChannels();
    }

  }}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpGsc16ai::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Gsc16ai::DataV1> data1 = evt.get(m_src);
  if (data1) {

    WithMsgLog(name(), info, str) {
      str << "Psana::Gsc16ai::DataV1:";
      str << "\n  timestamp = " << data1->timestamp();
      str << "\n  channelValue = " << data1->channelValue();
    }

  }
}

} // namespace psana_examples
