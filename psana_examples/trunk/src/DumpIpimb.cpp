//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpIpimb...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpIpimb.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/ipimb.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpIpimb)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpIpimb::DumpIpimb (const std::string& name)
  : Module(name)
{
  m_src = configStr("source", "DetInfo(:Ipimb)");
}

//--------------
// Destructor --
//--------------
DumpIpimb::~DumpIpimb ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpIpimb::beginCalibCycle(Env& env)
{
  MsgLog(name(), info, "in beginCalibCycle()");

  shared_ptr<Psana::Ipimb::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1.get()) {
    
    WithMsgLog(name(), info, str) {
      str << "Ipimb::ConfigV1:";
      str << "\n  triggerCounter = " << config1->triggerCounter();
      str << "\n  serialID = " << config1->serialID();
      str << "\n  chargeAmpRange = " << config1->chargeAmpRange();
      str << "\n  calibrationRange = " << config1->calibrationRange();
      str << "\n  resetLength = " << config1->resetLength();
      str << "\n  resetDelay = " << config1->resetDelay();
      str << "\n  chargeAmpRefVoltage = " << config1->chargeAmpRefVoltage();
      str << "\n  calibrationVoltage = " << config1->calibrationVoltage();
      str << "\n  diodeBias = " << config1->diodeBias();
      str << "\n  status = " << config1->status();
      str << "\n  errors = " << config1->errors();
      str << "\n  calStrobeLength = " << config1->calStrobeLength();
      str << "\n  trigDelay = " << config1->trigDelay();
    }
    
  }

}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpIpimb::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Ipimb::DataV1> data1 = evt.get(m_src);
  if (data1.get()) {
    
    WithMsgLog(name(), info, str) {
      str << "Ipimb::DataV1:"
          << "\n  triggerCounter = " << data1->triggerCounter()
          << "\n  config = " << data1->config0()
          << "," << data1->config1()
          << "," << data1->config2()
          << "\n  channel = " << data1->channel0()
          << "," << data1->channel1()
          << "," << data1->channel2()
          << "," << data1->channel3()
          << "\n  volts = " << data1->channel0Volts()
          << "," << data1->channel1Volts()
          << "," << data1->channel2Volts()
          << "," << data1->channel3Volts()
          << "\n  checksum = " << data1->checksum();
    }
  }
}
  
} // namespace psana_examples
