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
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ControlData::PVControl, name)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ControlData::PVControl, array)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ControlData::PVControl, index)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ControlData::PVControl, value)

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

  BaseType::initType( "PVControl", module );
}

void
pypdsdata::ControlData::PVControl::print(std::ostream& str) const
{
  str << "control.PVControl(name=" << m_obj.name();
  
  if (m_obj.array()) {
    str << ", index=" << m_obj.index();
  }

  str << ", value=" << m_obj.value();

  str << ")";
}
