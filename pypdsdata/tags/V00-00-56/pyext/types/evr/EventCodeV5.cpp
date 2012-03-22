//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_EventCodeV5...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "EventCodeV5.h"

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
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV5, code)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV5, desc)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV5, isReadout)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV5, isCommand)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV5, isLatch)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV5, reportDelay)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV5, reportWidth)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV5, releaseCode)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV5, maskTrigger)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV5, maskSet)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV5, maskClear)

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
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::EventCodeV5 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::EventCodeV5::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "EventCodeV5", module );
}
