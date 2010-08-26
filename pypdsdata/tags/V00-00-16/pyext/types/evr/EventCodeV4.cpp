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
    { "code",          code,         METH_NOARGS, "" },
    { "isReadout",     isReadout,    METH_NOARGS, "" },
    { "isTerminator",  isTerminator, METH_NOARGS, "" },
    { "reportDelay",   reportDelay,  METH_NOARGS, "" },
    { "reportWidth",   reportWidth,  METH_NOARGS, "" },
    { "maskTrigger",   maskTrigger,  METH_NOARGS, "" },
    { "maskSet",       maskSet,      METH_NOARGS, "" },
    { "maskClear",     maskClear,    METH_NOARGS, "" },
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
