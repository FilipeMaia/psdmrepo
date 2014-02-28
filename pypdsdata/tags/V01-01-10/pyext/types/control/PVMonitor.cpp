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
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ControlData::PVMonitor, name)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ControlData::PVMonitor, array)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ControlData::PVMonitor, index)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ControlData::PVMonitor, loValue)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ControlData::PVMonitor, hiValue)

  PyMethodDef methods[] = {
    {"name",       name,       METH_NOARGS,  "self.name() -> string\n\nReturns name of the monitoring channel" },
    {"array",      array,      METH_NOARGS,  "self.array() -> bool\n\nReturns true for array" },
    {"index",      index,      METH_NOARGS,  "self.index() -> int\n\nReturns index in the array" },
    {"loValue",    loValue,    METH_NOARGS,  "self.loValue() -> float\n\nReturns low value" },
    {"hiValue",    hiValue,    METH_NOARGS,  "self.hiValue() -> float\n\nReturns high value" },
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

void
pypdsdata::ControlData::PVMonitor::print(std::ostream& str) const
{
  str << "control.PVMonitor(name=" << m_obj.name();
  if (m_obj.array()) {
    str << ", index=" << m_obj.index();
  }
  str << ", loValue=" << m_obj.loValue()
      << ", hiValue=" << m_obj.hiValue()
      << ")";
}
