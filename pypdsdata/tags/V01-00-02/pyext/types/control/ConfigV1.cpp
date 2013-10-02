//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlData_ConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1.h"

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
#include "PVMonitor.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV1, uses_duration)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV1, uses_events)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV1, events)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV1, duration)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV1, npvControls)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV1, npvMonitors)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV1, pvControls)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV1, pvMonitors)
  PyObject* pvControl( PyObject* self, PyObject* args );
  PyObject* pvMonitor( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"uses_duration", uses_duration,  METH_NOARGS,  "self.uses_duration() -> bool\n\nReturns boolean value." },
    {"uses_events",   uses_events,    METH_NOARGS,  "self.uses_events() -> bool\n\nReturns boolean value." },
    {"duration",      duration,       METH_NOARGS,  "self.duration() -> xtc.Clocktime\n\nReturns :py:class:`_pdsdata.xtc.ClockTime` value." },
    {"events",        events,         METH_NOARGS,  "self.events() -> int\n\nReturns number of events." },
    {"npvControls",   npvControls,    METH_NOARGS,  "self.npvControls() -> int\n\nReturns number of PVControls." },
    {"npvMonitors",   npvMonitors,    METH_NOARGS,  "self.npvMonitors() -> int\n\nReturns number of PVMonitors." },
    {"pvControls",    pvControls,     METH_NOARGS,  "self.pvControls() -> list\n\nReturns list of :py:class:`PVControl` objects." },
    {"pvMonitors",    pvMonitors,     METH_NOARGS,  "self.pvMonitors() -> list\n\nReturns list of :py:class:`PVMonitor` objects." },
    {"pvControl",     pvControl,      METH_VARARGS, "self.pvControl(idx: int) -> control.PVControl\n\nReturns :py:class:`PVControl` for a given index." },
    {"pvMonitor",     pvMonitor,      METH_VARARGS, "self.pvMonitor(idx: int) -> control.PVMonitor\n\nReturns :py:class:`PVMonitor` for a given index." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::ControlData::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::ControlData::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::ControlData::ConfigV1::print(std::ostream& str) const
{
  str << "control.ConfigV1(";
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
  str << ")";
}

namespace {

PyObject*
pvControl( PyObject* self, PyObject* args )
{
  const Pds::ControlData::ConfigV1* obj = pypdsdata::ControlData::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ConfigV1_pvControl", &index ) ) return 0;

  return toPython(obj->pvControls()[index]);
}

PyObject*
pvMonitor( PyObject* self, PyObject* args )
{
  const Pds::ControlData::ConfigV1* obj = pypdsdata::ControlData::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ConfigV1_pvMonitor", &index ) ) return 0;

  return toPython(obj->pvMonitors()[index]);
}

}
