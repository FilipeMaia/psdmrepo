//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlData_ConfigV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV3.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../ClockTime.h"
#include "../../Exception.h"
#include "PVControl.h"
#include "PVLabel.h"
#include "PVMonitor.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV3, uses_l3t_events)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV3, uses_duration)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV3, uses_events)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV3, events)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV3, duration)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV3, npvControls)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV3, npvMonitors)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV3, npvLabels)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV3, pvControls)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV3, pvMonitors)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV3, pvLabels)
  PyObject* pvControl( PyObject* self, PyObject* args );
  PyObject* pvMonitor( PyObject* self, PyObject* args );
  PyObject* pvLabel( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"uses_l3t_events", uses_l3t_events,  METH_NOARGS,  "self.uses_l3t_events() -> bool\n\nReturns true if the configuration uses l3trigger events limit." },
    {"uses_duration", uses_duration,  METH_NOARGS,  "self.uses_duration() -> bool\n\nReturns true if the configuration uses duration control." },
    {"uses_events",   uses_events,    METH_NOARGS,  "self.uses_events() -> bool\n\nReturns true if the configuration uses events limit." },
    {"duration",      duration,       METH_NOARGS,  "self.duration() -> xtc.Clocktime\n\nReturns maximum duration of the scan, :py:class:`_pdsdata.xtc.ClockTime` value." },
    {"events",        events,         METH_NOARGS,  "self.events() -> int\n\nReturns maximum number of events per scan." },
    {"npvControls",   npvControls,    METH_NOARGS,  "self.npvControls() -> int\n\nReturns number of PVControls." },
    {"npvMonitors",   npvMonitors,    METH_NOARGS,  "self.npvMonitors() -> int\n\nReturns number of PVMonitors." },
    {"npvLabels",     npvLabels,      METH_NOARGS,  "self.npvLabels() -> int\n\nReturns number of PVLables." },
    {"pvControls",    pvControls,     METH_NOARGS,  "self.pvControls() -> list\n\nReturns list of :py:class:`PVControl` objects." },
    {"pvMonitors",    pvMonitors,     METH_NOARGS,  "self.pvMonitors() -> list\n\nReturns list of :py:class:`PVMonitor` objects." },
    {"pvLabels",      pvLabels,       METH_NOARGS,  "self.pvLabels() -> list\n\nReturns list of :py:class:`PVLabel` objects." },
    {"pvControl",     pvControl,      METH_VARARGS, "self.pvControl(idx: int) -> control.PVControl\n\nReturns :py:class:`PVControl` for a given index." },
    {"pvMonitor",     pvMonitor,      METH_VARARGS, "self.pvMonitor(idx: int) -> control.PVMonitor\n\nReturns :py:class:`PVMonitor` for a given index." },
    {"pvLabel",       pvLabel,        METH_VARARGS, "self.pvLabel(idx: int) -> control.PVLabel\n\nReturns :py:class:`PVLabel` for a given index." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::ControlData::ConfigV3 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::ControlData::ConfigV3::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV3", module );
}

void
pypdsdata::ControlData::ConfigV3::print(std::ostream& str) const
{
  str << "control.ConfigV3(";
  const char* comma = "";
  if (m_obj->uses_duration()) {
    const Pds::ClockTime& duration = m_obj->duration();
    double dur = duration.seconds() + duration.nanoseconds()/1e9;
    str << comma << "duration=" << dur << "sec";
    comma = ", ";
  }
  if (m_obj->uses_events()) {
    str << comma << "events=" << m_obj->events() ;
    comma = ", ";
  }
  if (m_obj->npvControls()) {
    str << comma << "controls=[";
    const ndarray<const Pds::ControlData::PVControl, 1>& pvControls = m_obj->pvControls();
    for (unsigned i = 0; i != pvControls.size(); ++ i ) {
      if (i != 0) str << ", ";
      const Pds::ControlData::PVControl& control = pvControls[i];
      str << control.name() << "=" << control.value();
    }
    str << "]";
    comma = ", ";
  }
  if (m_obj->npvMonitors()) {
    str << comma << "monitors=[";
    const ndarray<const Pds::ControlData::PVMonitor, 1>& pvMonitors = m_obj->pvMonitors();
    for (unsigned i = 0; i != pvMonitors.size(); ++ i ) {
      if (i != 0) str << ", ";
      const Pds::ControlData::PVMonitor& mon = pvMonitors[i];
      str << mon.name() << "=" << mon.loValue() << ":" << mon.hiValue();
    }
    str << "]";
    comma = ", ";
  }
  if (m_obj->npvLabels()) {
    str << comma << "labels=[";
    const ndarray<const Pds::ControlData::PVLabel, 1>& pvLabels = m_obj->pvLabels();
    for (unsigned i = 0; i != pvLabels.size(); ++ i ) {
      if (i != 0) str << ", ";
      const Pds::ControlData::PVLabel& lbl = pvLabels[i];
      str << lbl.name() << ": " << lbl.value();
    }
    str << "]";
    comma = ", ";
  }
  str << ")";
}

namespace {

PyObject*
pvControl( PyObject* self, PyObject* args )
{
  const Pds::ControlData::ConfigV3* obj = pypdsdata::ControlData::ConfigV3::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ConfigV3_pvControl", &index ) ) return 0;

  return toPython(obj->pvControls()[index]);
}

PyObject*
pvMonitor( PyObject* self, PyObject* args )
{
  const Pds::ControlData::ConfigV3* obj = pypdsdata::ControlData::ConfigV3::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ConfigV3_pvMonitor", &index ) ) return 0;

  return toPython(obj->pvMonitors()[index]);
}

PyObject*
pvLabel( PyObject* self, PyObject* args )
{
  const Pds::ControlData::ConfigV3* obj = pypdsdata::ControlData::ConfigV3::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ConfigV3_pvLabel", &index ) ) return 0;

  return toPython(obj->pvLabels()[index]);
}

}
