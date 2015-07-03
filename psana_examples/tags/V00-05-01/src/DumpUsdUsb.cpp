//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpUsdUsb...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpUsdUsb.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/usdusb.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpUsdUsb)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpUsdUsb::DumpUsdUsb (const std::string& name)
  : Module(name)
  , m_src()
{
  // get the values from configuration or use defaults
  m_src = configSrc("source", "DetInfo(:USDUSB)");
}

//--------------
// Destructor --
//--------------
DumpUsdUsb::~DumpUsdUsb ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpUsdUsb::beginCalibCycle(Event& evt, Env& env)
{
  shared_ptr<Psana::UsdUsb::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {
    WithMsgLog(name(), info, str) {
      str << "UsdUsb::ConfigV1:";
      str << "\n  counting_mode = " << config1->counting_mode();
      str << "\n  quadrature_mode = " << config1->quadrature_mode();
    }
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpUsdUsb::event(Event& evt, Env& env)
{
  shared_ptr<Psana::UsdUsb::DataV1> data1 = evt.get(m_src);
  if (data1) {
    WithMsgLog(name(), info, str) {
      str << "UsdUsb::DataV1:";
      str << "\n  encoder_count = " << data1->encoder_count();
      str << "\n  analog_in = " << data1->analog_in();
      ndarray<const uint8_t, 1> st = data1->status();
      str << "\n  status = [" << int(st[0]) << ' ' << int(st[1]) << ' ' << int(st[2]) << ' ' << int(st[3]) <<']' ;
      str << "\n  digital_in = " << int(data1->digital_in());
      str << "\n  timestamp = " << int(data1->timestamp());
    }
  }
}

} // namespace psana_examples
