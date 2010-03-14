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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ClockTime.h"
#include "Exception.h"
#include "PVControl.h"
#include "PVMonitor.h"
#include "types/TypeLib.h"

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
  PyObject* pvControl( PyObject* self, PyObject* );
  PyObject* pvMonitor( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"uses_duration", uses_duration,  METH_NOARGS,  "Returns boolean value." },
    {"uses_events",   uses_events,    METH_NOARGS,  "Returns boolean value." },
    {"duration",      duration,       METH_NOARGS,  "Returns ClockTime value." },
    {"events",        events,         METH_NOARGS,  "Returns number of events." },
    {"npvControls",   npvControls,    METH_NOARGS,  "Returns number of PVControls." },
    {"npvMonitors",   npvMonitors,    METH_NOARGS,  "Returns number of npvMonitors." },
    {"size",          size,           METH_NOARGS,  "Returns total data size." },
    {"pvControl",     pvControl,      METH_VARARGS, "Returns PVControl for a given index." },
    {"pvMonitor",     pvMonitor,      METH_VARARGS, "Returns pvMonitor for a given index." },
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

}
