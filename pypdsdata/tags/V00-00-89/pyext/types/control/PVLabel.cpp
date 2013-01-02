//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlData_PVLabel...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PVLabel.h"

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
  FUN0_WRAPPER(pypdsdata::ControlData::PVLabel, name)
  FUN0_WRAPPER(pypdsdata::ControlData::PVLabel, value)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"name",       name,       METH_NOARGS,  "self.name() -> string\n\nReturns name of the monitoring channel" },
    {"value",      value,      METH_NOARGS,  "self.value() -> string\n\nReturns label string" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::ControlData::PVLabel class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::ControlData::PVLabel::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "PVLabel", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::ControlData::PVLabel* obj = pypdsdata::ControlData::PVLabel::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "control.PVLabel(name=" << obj->name();
  str << ", value=" << obj->value();
  str << ")";
  
  return PyString_FromString( str.str().c_str() );
}

}
