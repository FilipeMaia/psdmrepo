//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlData_PVMonitor...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PVMonitor.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Exception.h"
#include "types/TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::ControlData::PVMonitor, name)
  FUN0_WRAPPER(pypdsdata::ControlData::PVMonitor, array)
  FUN0_WRAPPER(pypdsdata::ControlData::PVMonitor, index)
  FUN0_WRAPPER(pypdsdata::ControlData::PVMonitor, loValue)
  FUN0_WRAPPER(pypdsdata::ControlData::PVMonitor, hiValue)

  PyMethodDef methods[] = {
    {"name",       name,       METH_NOARGS,  "Returns name of the monitoring channel" },
    {"array",      array,      METH_NOARGS,  "Returns true for array" },
    {"index",      index,      METH_NOARGS,  "Returns index in the array" },
    {"loValue",    loValue,    METH_NOARGS,  "Returns low value" },
    {"hiValue",    hiValue,    METH_NOARGS,  "Returns high value" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::ControlData::PVMonitor class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::ControlData::PVMonitor::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "PVMonitor", module );
}
