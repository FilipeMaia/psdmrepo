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


  void printPVControl(std::ostream& str, const Psana::ControlData::PVControl& ctrl)
  {
    str << "\n    " << ctrl.name() << " index=" << ctrl.index()
        << " value=" << ctrl.value() << " array=" << int(ctrl.array());
  }

  void printPVMonitor(std::ostream& str, const Psana::ControlData::PVMonitor& mon)
  {
    str << "\n    " << mon.name() << " index=" << mon.index()
        << " low value=" << mon.loValue()
        << " high value=" << mon.hiValue()
        << " array=" << int(mon.array());
  }

  void printPVLabel(std::ostream& str, const Psana::ControlData::PVLabel& lbl)
  {
    str << "\n    " << lbl.name() << " value=" << lbl.value();
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
  if (config) {
    
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
        printPVControl(str, pvControls[i]);
      }

      const ndarray<const Psana::ControlData::PVMonitor, 1>& pvMonitors = config->pvMonitors();
      for (unsigned i = 0; i < pvMonitors.size(); ++ i) {
        if (i == 0) str << "\n  PV Monitors:";
        printPVMonitor(str, pvMonitors[i]);
      }
    }
    
  }


  shared_ptr<Psana::ControlData::ConfigV2> config2 = env.configStore().get(m_src);
  if (config2) {

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
        printPVControl(str, pvControls[i]);
      }

      const ndarray<const Psana::ControlData::PVMonitor, 1>& pvMonitors = config2->pvMonitors();
      for (unsigned i = 0; i < pvMonitors.size(); ++ i) {
        if (i == 0) str << "\n  PV Monitors:";
        printPVMonitor(str, pvMonitors[i]);
      }

      const ndarray<const Psana::ControlData::PVLabel, 1>& pvLabels = config2->pvLabels();
      for (unsigned i = 0; i < pvLabels.size(); ++ i) {
        if (i == 0) str << "\n  PV Labels:";
        printPVLabel(str, pvLabels[i]);
      }
    }

  }


  shared_ptr<Psana::ControlData::ConfigV3> config3 = env.configStore().get(m_src);
  if (config3) {

    WithMsgLog(name(), info, str) {

      str << "ControlData::ConfigV3:"
          <<  "\n  uses_duration = " << (config3->uses_duration() ? "yes" : "no")
          <<  "\n  duration = ";
      printClockTime(str, config3->duration());
      str <<  "\n  uses_events = " << (config3->uses_events() ? "yes" : "no")
          <<  "\n  uses_l3t_events = " << (config3->uses_l3t_events() ? "yes" : "no")
          << "\n  events = " << config3->events();

      const ndarray<const Psana::ControlData::PVControl, 1>& pvControls = config3->pvControls();
      for (unsigned i = 0; i < pvControls.size(); ++ i) {
        if (i == 0) str << "\n  PV Controls:";
        printPVControl(str, pvControls[i]);
      }

      const ndarray<const Psana::ControlData::PVMonitor, 1>& pvMonitors = config3->pvMonitors();
      for (unsigned i = 0; i < pvMonitors.size(); ++ i) {
        if (i == 0) str << "\n  PV Monitors:";
        printPVMonitor(str, pvMonitors[i]);
      }

      const ndarray<const Psana::ControlData::PVLabel, 1>& pvLabels = config3->pvLabels();
      for (unsigned i = 0; i < pvLabels.size(); ++ i) {
        if (i == 0) str << "\n  PV Labels:";
        printPVLabel(str, pvLabels[i]);
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
