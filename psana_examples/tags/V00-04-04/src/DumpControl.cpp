//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpControl...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpControl.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iomanip>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/control.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpControl)

namespace {
  
  void
  printClockTime(std::ostream& str, const Pds::ClockTime& time) 
  {
    str << time.seconds();
    if (time.nanoseconds()) {
      str.fill('0');
      str << std::setw(9) << time.nanoseconds();
      str.fill(' ');
    }
    str << "sec";
  }
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpControl::DumpControl (const std::string& name)
  : Module(name)
{
  m_src = configSrc("source", "ProcInfo()");
}

//--------------
// Destructor --
//--------------
DumpControl::~DumpControl ()
{
}

// Method which is called at the beginning of the calibration cycle
void 
DumpControl::beginCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), trace, "in beginCalibCycle()");

  shared_ptr<Psana::ControlData::ConfigV1> config = env.configStore().get(m_src);
  if (not config) {
    MsgLog(name(), info, "ControlData::ConfigV1 not found");    
  } else {
    
    WithMsgLog(name(), info, str) {
      
      str << "ControlData::ConfigV1:"
          <<  "\n  uses_duration = " << (config->uses_duration() ? "yes" : "no")
          <<  "\n  duration = ";
      printClockTime(str, config->duration());
      str <<  "\n  uses_events = " << (config->uses_events() ? "yes" : "no")
          << "\n  events = " << config->events();

      const ndarray<const Psana::ControlData::PVControl, 1>& pvControls = config->pvControls();
      for (unsigned i = 0; i < pvControls.size(); ++ i) {
        if (i == 0) str << "\n  PV Controls:";
        const Psana::ControlData::PVControl& ctrl = pvControls[i];
        str << "\n    " << ctrl.name() << " index=" << ctrl.index()
            << " value=" << ctrl.value() << " array=" << int(ctrl.array());
        
      }

      const ndarray<const Psana::ControlData::PVMonitor, 1>& pvMonitors = config->pvMonitors();
      for (unsigned i = 0; i < pvMonitors.size(); ++ i) {
        if (i == 0) str << "\n  PV Monitors:";
        const Psana::ControlData::PVMonitor& mon = pvMonitors[i];
        str << "\n    " << mon.name() << " index=" << mon.index()
            << " low value=" << mon.loValue() 
            << " high value=" << mon.hiValue() 
            << " array=" << int(mon.array());
        
      }
    }
    
  }


  shared_ptr<Psana::ControlData::ConfigV2> config2 = env.configStore().get(m_src);
  if (not config2) {
    MsgLog(name(), info, "ControlData::ConfigV2 not found");
  } else {

    WithMsgLog(name(), info, str) {

      str << "ControlData::ConfigV2:"
          <<  "\n  uses_duration = " << (config2->uses_duration() ? "yes" : "no")
          <<  "\n  duration = ";
      printClockTime(str, config2->duration());
      str <<  "\n  uses_events = " << (config2->uses_events() ? "yes" : "no")
          << "\n  events = " << config2->events();

      const ndarray<const Psana::ControlData::PVControl, 1>& pvControls = config2->pvControls();
      for (unsigned i = 0; i < pvControls.size(); ++ i) {
        if (i == 0) str << "\n  PV Controls:";
        const Psana::ControlData::PVControl& ctrl = pvControls[i];
        str << "\n    " << ctrl.name() << " index=" << ctrl.index()
            << " value=" << ctrl.value() << " array=" << int(ctrl.array());

      }

      const ndarray<const Psana::ControlData::PVMonitor, 1>& pvMonitors = config2->pvMonitors();
      for (unsigned i = 0; i < pvMonitors.size(); ++ i) {
        if (i == 0) str << "\n  PV Monitors:";
        const Psana::ControlData::PVMonitor& mon = pvMonitors[i];
        str << "\n    " << mon.name() << " index=" << mon.index()
            << " low value=" << mon.loValue()
            << " high value=" << mon.hiValue()
            << " array=" << int(mon.array());

      }

      const ndarray<const Psana::ControlData::PVLabel, 1>& pvLabels = config2->pvLabels();
      for (unsigned i = 0; i < pvLabels.size(); ++ i) {
        if (i == 0) str << "\n  PV Labels:";
        const Psana::ControlData::PVLabel& lbl = pvLabels[i];
        str << "\n    " << lbl.name() << " value=" << lbl.value();
      }
    }

  }
}

// Method which is called with event data
void 
DumpControl::event(Event& evt, Env& env)
{
}
  
} // namespace psana_examples
