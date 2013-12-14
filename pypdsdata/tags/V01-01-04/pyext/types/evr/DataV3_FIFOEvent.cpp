//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_DataV3_FIFOEvent...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataV3_FIFOEvent.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  namespace gs {
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::EvrData::DataV3_FIFOEvent, timestampHigh)
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::EvrData::DataV3_FIFOEvent, timestampLow)
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::EvrData::DataV3_FIFOEvent, eventCode)
  }

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"TimestampHigh",  gs::timestampHigh,     0, "Integer number", 0},
    {"TimestampLow",   gs::timestampLow,      0, "Integer number", 0},
    {"EventCode",      gs::eventCode,         0, "Integer number", 0},
    {0, 0, 0, 0, 0}
  };

  namespace mm {
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::DataV3_FIFOEvent, timestampHigh)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::DataV3_FIFOEvent, timestampLow)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::DataV3_FIFOEvent, eventCode)
  }

  PyMethodDef methods[] = {
    { "timestampHigh",   mm::timestampHigh,  METH_NOARGS, "self.timestampHigh() -> int\n\nReturns integer number" },
    { "timestampLow",    mm::timestampLow,   METH_NOARGS, "self.timestampLow() -> int\n\nReturns integer number" },
    { "eventCode",       mm::eventCode,      METH_NOARGS, "self.eventCode() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::DataV3::FIFOEvent class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::DataV3_FIFOEvent::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_methods = ::methods;

  BaseType::initType( "DataV3_FIFOEvent", module );
}
