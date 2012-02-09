//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_EventCodeV4...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "EventCodeV4.h"

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
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV4, code)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV4, isReadout)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV4, isTerminator)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV4, reportDelay)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV4, reportWidth)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV4, maskTrigger)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV4, maskSet)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV4, maskClear)

  PyMethodDef methods[] = {
    { "code",          code,         METH_NOARGS, "self.code() -> int\n\nReturns integer number" },
    { "isReadout",     isReadout,    METH_NOARGS, "self.isReadout() -> bool\n\nReturns boolean" },
    { "isTerminator",  isTerminator, METH_NOARGS, "self.isTerminator() -> bool\n\nReturns boolean" },
    { "reportDelay",   reportDelay,  METH_NOARGS, "self.reportDelay() -> int\n\nReturns integer number" },
    { "reportWidth",   reportWidth,  METH_NOARGS, "self.reportWidth() -> int\n\nReturns integer number" },
    { "maskTrigger",   maskTrigger,  METH_NOARGS, "self.maskTrigger() -> int\n\nReturns integer number" },
    { "maskSet",       maskSet,      METH_NOARGS, "self.maskSet() -> int\n\nReturns integer number" },
    { "maskClear",     maskClear,    METH_NOARGS, "self.maskClear() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::EventCodeV4 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::EventCodeV4::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "EventCodeV4", module );
}
