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
    { "code",          code,         METH_NOARGS, "" },
    { "desc",          desc,         METH_NOARGS, "" },
    { "isReadout",     isReadout,    METH_NOARGS, "" },
    { "isCommand",     isCommand,    METH_NOARGS, "" },
    { "isLatch",       isLatch,      METH_NOARGS, "" },
    { "reportDelay",   reportDelay,  METH_NOARGS, "" },
    { "reportWidth",   reportWidth,  METH_NOARGS, "" },
    { "releaseCode",   releaseCode,  METH_NOARGS, "" },
    { "maskTrigger",   maskTrigger,  METH_NOARGS, "" },
    { "maskSet",       maskSet,      METH_NOARGS, "" },
    { "maskClear",     maskClear,    METH_NOARGS, "" },
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
