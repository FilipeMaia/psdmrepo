//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_EventCodeV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "EventCodeV3.h"

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
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV3, code)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV3, isReadout)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV3, isTerminator)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV3, maskTrigger)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV3, maskSet)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::EventCodeV3, maskClear)

  PyMethodDef methods[] = {
    { "code",          code,         METH_NOARGS, "self.code() -> int\n\nReturns integer number" },
    { "isReadout",     isReadout,    METH_NOARGS, "self.isReadout() -> bool\n\nReturns boolean" },
    { "isTerminator",  isTerminator, METH_NOARGS, "self.isTerminator() -> bool\n\nReturns boolean" },
    { "maskTrigger",   maskTrigger,  METH_NOARGS, "self.maskTrigger() -> int\n\nReturns integer number" },
    { "maskSet",       maskSet,      METH_NOARGS, "self.maskSet() -> int\n\nReturns integer number" },
    { "maskClear",     maskClear,    METH_NOARGS, "self.maskClear() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::EventCodeV3 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::EventCodeV3::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "EventCodeV3", module );
}
