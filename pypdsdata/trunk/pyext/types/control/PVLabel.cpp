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
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ControlData::PVLabel, name)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ControlData::PVLabel, value)

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

  BaseType::initType( "PVLabel", module );
}

void
pypdsdata::ControlData::PVLabel::print(std::ostream& str) const
{
  str << "control.PVLabel(name=" << m_obj.name();
  str << ", value=" << m_obj.value();
  str << ")";
}
