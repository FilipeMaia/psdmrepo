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

  PyMethodDef methods[] = {
    {"name",       name,       METH_NOARGS,  "Returns name of the monitoring channel" },
    {"array",      array,      METH_NOARGS,  "Returns true for array" },
    {"index",      index,      METH_NOARGS,  "Returns index in the array" },
    {"value",      value,      METH_NOARGS,  "Returns value" },
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
