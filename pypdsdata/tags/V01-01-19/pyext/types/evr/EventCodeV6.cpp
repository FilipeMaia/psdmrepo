//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_EventCodeV6...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "EventCodeV6.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, code)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, desc)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, isReadout)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, isCommand)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, isLatch)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, reportDelay)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, reportWidth)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, releaseCode)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, maskTrigger)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, maskSet)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, maskClear)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV6, readoutGroup)

  PyMethodDef methods[] = {
    { "code",          code,         METH_NOARGS, "self.code() -> int\n\nReturns integer number" },
    { "desc",          desc,         METH_NOARGS, "self.desc() -> string\n\nReturns string" },
    { "isReadout",     isReadout,    METH_NOARGS, "self.isReadout() -> bool\n\nReturns boolean" },
    { "isCommand",     isCommand,    METH_NOARGS, "self.isCommand() -> bool\n\nReturns boolean" },
    { "isLatch",       isLatch,      METH_NOARGS, "self.isLatch() -> bool\n\nReturns boolean" },
    { "reportDelay",   reportDelay,  METH_NOARGS, "self.reportDelay() -> int\n\nReturns integer number" },
    { "reportWidth",   reportWidth,  METH_NOARGS, "self.reportWidth() -> int\n\nReturns integer number" },
    { "releaseCode",   releaseCode,  METH_NOARGS, "self.releaseCode() -> int\n\nReturns integer number" },
    { "maskTrigger",   maskTrigger,  METH_NOARGS, "self.maskTrigger() -> int\n\nReturns integer number" },
    { "maskSet",       maskSet,      METH_NOARGS, "self.maskSet() -> int\n\nReturns integer number" },
    { "maskClear",     maskClear,    METH_NOARGS, "self.maskClear() -> int\n\nReturns integer number" },
    { "readoutGroup",  readoutGroup, METH_NOARGS, "self.readoutGroup() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::EventCodeV6 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::EventCodeV6::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::EvrData::EventCodeV6::MaxReadoutGroup);
  PyDict_SetItemString( type->tp_dict, "MaxReadoutGroup", val );
  Py_XDECREF(val);

  BaseType::initType( "EventCodeV6", module );
}
