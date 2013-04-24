//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlData_ConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV2.h"

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
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV2, uses_duration)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV2, uses_events)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV2, events)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV2, npvControls)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV2, npvMonitors)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV2, npvLabels)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV2, size)
  PyObject* duration( PyObject* self, PyObject* );
  PyObject* pvControl( PyObject* self, PyObject* args );
  PyObject* pvMonitor( PyObject* self, PyObject* args );
  PyObject* pvLabel( PyObject* self, PyObject* args );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"uses_duration", uses_duration,  METH_NOARGS,  "self.uses_duration() -> bool\n\nReturns boolean value." },
    {"uses_events",   uses_events,    METH_NOARGS,  "self.uses_events() -> bool\n\nReturns boolean value." },
    {"duration",      duration,       METH_NOARGS,  "self.duration() -> xtc.Clocktime\n\nReturns :py:class:`_pdsdata.xtc.ClockTime` value." },
    {"events",        events,         METH_NOARGS,  "self.events() -> int\n\nReturns number of events." },
    {"npvControls",   npvControls,    METH_NOARGS,  "self.npvControls() -> int\n\nReturns number of PVControls." },
    {"npvMonitors",   npvMonitors,    METH_NOARGS,  "self.npvMonitors() -> int\n\nReturns number of PVMonitors." },
    {"npvLabels",     npvLabels,      METH_NOARGS,  "self.npvLabels() -> int\n\nReturns number of PVLables." },
    {"size",          size,           METH_NOARGS,  "self.size() -> int\n\nReturns total data size." },
    {"pvControl",     pvControl,      METH_VARARGS, "self.pvControl(idx: int) -> control.PVControl\n\nReturns :py:class:`PVControl` for a given index." },
    {"pvMonitor",     pvMonitor,      METH_VARARGS, "self.pvMonitor(idx: int) -> control.PVMonitor\n\nReturns :py:class:`PVMonitor` for a given index." },
    {"pvLabel",       pvLabel,        METH_VARARGS, "self.pvLabel(idx: int) -> control.PVLabel\n\nReturns :py:class:`PVLabel` for a given index." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::ControlData::ConfigV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::ControlData::ConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "ConfigV2", module );
}

namespace {

PyObject*
duration( PyObject* self, PyObject* args)
{
  const Pds::ControlData::ConfigV2* obj = pypdsdata::ControlData::ConfigV2::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::ClockTime::PyObject_FromPds( obj->duration() );
}

PyObject*
pvControl( PyObject* self, PyObject* args )
{
  const Pds::ControlData::ConfigV2* obj = pypdsdata::ControlData::ConfigV2::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ConfigV2_pvControl", &index ) ) return 0;

  return pypdsdata::ControlData::PVControl::PyObject_FromPds( (Pds::ControlData::PVControl*)(&obj->pvControl(index)),
      self, sizeof(Pds::ControlData::PVControl) );
}

PyObject*
pvMonitor( PyObject* self, PyObject* args )
{
  const Pds::ControlData::ConfigV2* obj = pypdsdata::ControlData::ConfigV2::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ConfigV2_pvMonitor", &index ) ) return 0;

  return pypdsdata::ControlData::PVMonitor::PyObject_FromPds( (Pds::ControlData::PVMonitor*)(&obj->pvMonitor(index)),
      self, sizeof(Pds::ControlData::PVMonitor) );
}

PyObject*
pvLabel( PyObject* self, PyObject* args )
{
  const Pds::ControlData::ConfigV2* obj = pypdsdata::ControlData::ConfigV2::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ConfigV2_pvLabel", &index ) ) return 0;

  return pypdsdata::ControlData::PVLabel::PyObject_FromPds( (Pds::ControlData::PVLabel*)(&obj->pvLabel(index)),
      self, sizeof(Pds::ControlData::PVLabel) );
}

PyObject*
_repr( PyObject *self )
{
  Pds::ControlData::ConfigV2* pdsObj = pypdsdata::ControlData::ConfigV2::pdsObject(self);
  if(not pdsObj) return 0;

  std::ostringstream str;
  str << "control.ConfigV2(";
  const char* comma = "";
  if (pdsObj->uses_duration()) {
    const ClockTime& duration = pdsObj->duration();
    double dur = duration.seconds() + duration.nanoseconds()/1e9;
    str << comma << "duration=" << dur << "sec";
    comma = ", ";
  }
  if (pdsObj->uses_events()) {
    str << comma << "events=" << pdsObj->events() ;
    comma = ", ";
  }
  if (pdsObj->npvControls()) {
    str << comma << "controls=["; 
    for (unsigned i = 0; i != pdsObj->npvControls(); ++ i ) {
      if (i != 0) str << ", ";
      str << pdsObj->pvControl(i).name();
    }
    str << "]";
    comma = ", ";
  }
  if (pdsObj->npvMonitors()) {
    str << comma << "monitors=["; 
    for (unsigned i = 0; i != pdsObj->npvMonitors(); ++ i ) {
      if (i != 0) str << ", ";
      str << pdsObj->pvMonitor(i).name();
    }
    str << "]";
    comma = ", ";
  }
  if (pdsObj->npvLabels()) {
    str << comma << "labels=[";
    for (unsigned i = 0; i != pdsObj->npvLabels(); ++ i ) {
      if (i != 0) str << ", ";
      str << pdsObj->pvLabel(i).name() << ": " << pdsObj->pvLabel(i).value();
    }
    str << "]";
    comma = ", ";
  }
  str << ")";
  return PyString_FromString( str.str().c_str() );
}

}
