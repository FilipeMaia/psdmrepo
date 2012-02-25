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
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV1, npvControls)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV1, npvMonitors)
  FUN0_WRAPPER(pypdsdata::ControlData::ConfigV1, size)
  PyObject* duration( PyObject* self, PyObject* );
  PyObject* pvControl( PyObject* self, PyObject* args );
  PyObject* pvMonitor( PyObject* self, PyObject* args );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"uses_duration", uses_duration,  METH_NOARGS,  "self.uses_duration() -> bool\n\nReturns boolean value." },
    {"uses_events",   uses_events,    METH_NOARGS,  "self.uses_events() -> bool\n\nReturns boolean value." },
    {"duration",      duration,       METH_NOARGS,  "self.duration() -> xtc.Clocktime\n\nReturns ClockTime value." },
    {"events",        events,         METH_NOARGS,  "self.events() -> int\n\nReturns number of events." },
    {"npvControls",   npvControls,    METH_NOARGS,  "self.npvControls() -> int\n\nReturns number of PVControls." },
    {"npvMonitors",   npvMonitors,    METH_NOARGS,  "self.npvMonitors() -> int\n\nReturns number of PVMonitors." },
    {"size",          size,           METH_NOARGS,  "self.size() -> int\n\nReturns total data size." },
    {"pvControl",     pvControl,      METH_VARARGS, "self.pvControl(idx: int) -> control.PVControl\n\nReturns PVControl for a given index." },
    {"pvMonitor",     pvMonitor,      METH_VARARGS, "self.pvMonitor(idx: int) -> control.PVMonitor\n\nReturns PVMonitor for a given index." },
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "ConfigV1", module );
}

namespace {

PyObject*
duration( PyObject* self, PyObject* args)
{
  const Pds::ControlData::ConfigV1* obj = pypdsdata::ControlData::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  return pypdsdata::ClockTime::PyObject_FromPds( obj->duration() );
}

PyObject*
pvControl( PyObject* self, PyObject* args )
{
  const Pds::ControlData::ConfigV1* obj = pypdsdata::ControlData::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ConfigV1_pvControl", &index ) ) return 0;

  return pypdsdata::ControlData::PVControl::PyObject_FromPds( (Pds::ControlData::PVControl*)(&obj->pvControl(index)),
      self, sizeof(Pds::ControlData::PVControl) );
}

PyObject*
pvMonitor( PyObject* self, PyObject* args )
{
  const Pds::ControlData::ConfigV1* obj = pypdsdata::ControlData::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:ConfigV1_pvMonitor", &index ) ) return 0;

  return pypdsdata::ControlData::PVMonitor::PyObject_FromPds( (Pds::ControlData::PVMonitor*)(&obj->pvMonitor(index)),
      self, sizeof(Pds::ControlData::PVMonitor) );
}

PyObject*
_repr( PyObject *self )
{
  Pds::ControlData::ConfigV1* pdsObj = pypdsdata::ControlData::ConfigV1::pdsObject(self);
  if(not pdsObj) return 0;

  std::ostringstream str;
  str << "control.ConfigV1(";
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
  str << ")";
  return PyString_FromString( str.str().c_str() );
}

}
