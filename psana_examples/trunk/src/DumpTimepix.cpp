//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpTimepix...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpTimepix.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
#include "psddl_psana/timepix.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpTimepix)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpTimepix::DumpTimepix (const std::string& name)
  : Module(name)
  , m_src()
{
  // get the values from configuration or use defaults
  m_src = configStr("source", "DetInfo(:Timepix)");
}

//--------------
// Destructor --
//--------------
DumpTimepix::~DumpTimepix ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpTimepix::beginCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), trace, "in beginCalibCycle()");

  shared_ptr<Psana::Timepix::ConfigV1> config = env.configStore().get(m_src);
  if (config.get()) {

    WithMsgLog(name(), info, str) {
      str << "Timepix::ConfigV1:";

      str << "\n  readoutSpeed = " << int(config->readoutSpeed());
      str << "\n  triggerMode = " << int(config->triggerMode());
      str << "\n  shutterTimeout = " << config->shutterTimeout();
      str << "\n  dac0Ikrum = " << config->dac0Ikrum();
      str << "\n  dac0Disc = " << config->dac0Disc();
      str << "\n  dac0Preamp = " << config->dac0Preamp();
      str << "\n  dac0BufAnalogA = " << config->dac0BufAnalogA();
      str << "\n  dac0BufAnalogB = " << config->dac0BufAnalogB();
      str << "\n  dac0Hist = " << config->dac0Hist();
      str << "\n  dac0ThlFine = " << config->dac0ThlFine();
      str << "\n  dac0ThlCourse = " << config->dac0ThlCourse();
      str << "\n  dac0Vcas = " << config->dac0Vcas();
      str << "\n  dac0Fbk = " << config->dac0Fbk();
      str << "\n  dac0Gnd = " << config->dac0Gnd();
      str << "\n  dac0Ths = " << config->dac0Ths();
      str << "\n  dac0BiasLvds = " << config->dac0BiasLvds();
      str << "\n  dac0RefLvds = " << config->dac0RefLvds();
      str << "\n  dac1Ikrum = " << config->dac1Ikrum();
      str << "\n  dac1Disc = " << config->dac1Disc();
      str << "\n  dac1Preamp = " << config->dac1Preamp();
      str << "\n  dac1BufAnalogA = " << config->dac1BufAnalogA();
      str << "\n  dac1BufAnalogB = " << config->dac1BufAnalogB();
      str << "\n  dac1Hist = " << config->dac1Hist();
      str << "\n  dac1ThlFine = " << config->dac1ThlFine();
      str << "\n  dac1ThlCourse = " << config->dac1ThlCourse();
      str << "\n  dac1Vcas = " << config->dac1Vcas();
      str << "\n  dac1Fbk = " << config->dac1Fbk();
      str << "\n  dac1Gnd = " << config->dac1Gnd();
      str << "\n  dac1Ths = " << config->dac1Ths();
      str << "\n  dac1BiasLvds = " << config->dac1BiasLvds();
      str << "\n  dac1RefLvds = " << config->dac1RefLvds();
      str << "\n  dac2Ikrum = " << config->dac2Ikrum();
      str << "\n  dac2Disc = " << config->dac2Disc();
      str << "\n  dac2Preamp = " << config->dac2Preamp();
      str << "\n  dac2BufAnalogA = " << config->dac2BufAnalogA();
      str << "\n  dac2BufAnalogB = " << config->dac2BufAnalogB();
      str << "\n  dac2Hist = " << config->dac2Hist();
      str << "\n  dac2ThlFine = " << config->dac2ThlFine();
      str << "\n  dac2ThlCourse = " << config->dac2ThlCourse();
      str << "\n  dac2Vcas = " << config->dac2Vcas();
      str << "\n  dac2Fbk = " << config->dac2Fbk();
      str << "\n  dac2Gnd = " << config->dac2Gnd();
      str << "\n  dac2Ths = " << config->dac2Ths();
      str << "\n  dac2BiasLvds = " << config->dac2BiasLvds();
      str << "\n  dac2RefLvds = " << config->dac2RefLvds();
      str << "\n  dac3Ikrum = " << config->dac3Ikrum();
      str << "\n  dac3Disc = " << config->dac3Disc();
      str << "\n  dac3Preamp = " << config->dac3Preamp();
      str << "\n  dac3BufAnalogA = " << config->dac3BufAnalogA();
      str << "\n  dac3BufAnalogB = " << config->dac3BufAnalogB();
      str << "\n  dac3Hist = " << config->dac3Hist();
      str << "\n  dac3ThlFine = " << config->dac3ThlFine();
      str << "\n  dac3ThlCourse = " << config->dac3ThlCourse();
      str << "\n  dac3Vcas = " << config->dac3Vcas();
      str << "\n  dac3Fbk = " << config->dac3Fbk();
      str << "\n  dac3Gnd = " << config->dac3Gnd();
      str << "\n  dac3Ths = " << config->dac3Ths();
      str << "\n  dac3BiasLvds = " << config->dac3BiasLvds();
      str << "\n  dac3RefLvds = " << config->dac3RefLvds();
    }

  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpTimepix::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Timepix::DataV1> data1 = evt.get(m_src);
  if (data1.get()) {
    WithMsgLog(name(), info, str) {
      str << "Timepix::DataV1:";

      str << "\n  timestamp = " << data1->timestamp();
      str << "\n  frameCounter = " << data1->frameCounter();
      str << "\n  lostRows = " << data1->lostRows();

      const ndarray<uint16_t, 2>& img = data1->data();
      str << "\n  data =";
      str << " (" << img.shape()[0] << ", " << img.shape()[0] << ")";
      for (int i = 0; i < 10; ++ i) {
        str << " " << img[0][i];
      }
      str << " ...";
    }
  }

  shared_ptr<Psana::Timepix::DataV2> data2 = evt.get(m_src);
  if (data2.get()) {
    WithMsgLog(name(), info, str) {
      str << "Timepix::DataV2:";

      str << "\n  timestamp = " << data2->timestamp();
      str << "\n  frameCounter = " << data2->frameCounter();
      str << "\n  lostRows = " << data2->lostRows();

      const ndarray<uint16_t, 2>& img = data2->data();
      str << "\n  data =";
      str << " (" << img.shape()[0] << ", " << img.shape()[0] << ")";
      for (int i = 0; i < 10; ++ i) {
        str << " " << img[0][i];
      }
      str << " ...";
    }
  }
}
  

} // namespace psana_examples
