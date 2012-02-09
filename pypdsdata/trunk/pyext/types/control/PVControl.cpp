//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlData_PVControl...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PVControl.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::ControlData::PVControl, name)
  FUN0_WRAPPER(pypdsdata::ControlData::PVControl, array)
  FUN0_WRAPPER(pypdsdata::ControlData::PVControl, index)
  FUN0_WRAPPER(pypdsdata::ControlData::PVControl, value)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"name",       name,       METH_NOARGS,  "self.name() -> string\n\nReturns name of the monitoring channel" },
    {"array",      array,      METH_NOARGS,  "self.array() -> bool\n\nReturns true for array" },
    {"index",      index,      METH_NOARGS,  "self.index() -> int\n\nReturns index in the array" },
    {"value",      value,      METH_NOARGS,  "self.value() -> float\n\nReturns value as floating point number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::ControlData::PVControl class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::ControlData::PVControl::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "PVControl", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::ControlData::PVControl* obj = pypdsdata::ControlData::PVControl::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "control.PVControl(name=" << obj->name();
  
  if (obj->array()) {
    str << ", index=" << obj->index();
  }

  str << ", value=" << obj->value();

  str << ")";
  
  return PyString_FromString( str.str().c_str() );
}

}
