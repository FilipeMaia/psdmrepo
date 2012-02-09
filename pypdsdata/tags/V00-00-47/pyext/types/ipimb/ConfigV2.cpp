//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV2.h"

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

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, triggerCounter)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, serialID)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, chargeAmpRange)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, calibrationRange)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, resetLength)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, resetDelay)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, chargeAmpRefVoltage)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, calibrationVoltage)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, diodeBias)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, status)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, errors)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, calStrobeLength)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, trigDelay)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, trigPsDelay)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV2, adcDelay)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "triggerCounter",      triggerCounter,      METH_NOARGS, "self.triggerCounter() -> int\n\nReturns integer number" },
    { "serialID",            serialID,            METH_NOARGS, "self.serialID() -> int\n\nReturns integer number" },
    { "chargeAmpRange",      chargeAmpRange,      METH_NOARGS, "self.chargeAmpRange() -> int\n\nReturns integer number" },
    { "calibrationRange",    calibrationRange,    METH_NOARGS, "self.calibrationRange() -> int\n\nReturns integer number" },
    { "resetLength",         resetLength,         METH_NOARGS, "self.resetLength() -> int\n\nReturns integer number" },
    { "resetDelay",          resetDelay,          METH_NOARGS, "self.resetDelay() -> int\n\nReturns integer number" },
    { "chargeAmpRefVoltage", chargeAmpRefVoltage, METH_NOARGS, "self.chargeAmpRefVoltage() -> float\n\nReturns floating number" },
    { "calibrationVoltage",  calibrationVoltage,  METH_NOARGS, "self.calibrationVoltage() -> float\n\nReturns floating number" },
    { "diodeBias",           diodeBias,           METH_NOARGS, "self.diodeBias() -> float\n\nReturns floating number" },
    { "status",              status,              METH_NOARGS, "self.status() -> int\n\nReturns integer number" },
    { "errors",              errors,              METH_NOARGS, "self.errors() -> int\n\nReturns integer number" },
    { "calStrobeLength",     calStrobeLength,     METH_NOARGS, "self.calStrobeLength() -> int\n\nReturns integer number" },
    { "trigDelay",           trigDelay,           METH_NOARGS, "self.trigDelay() -> int\n\nReturns integer number" },
    { "trigPsDelay",         trigPsDelay,         METH_NOARGS, "self.trigPsDelay() -> int\n\nReturns integer number" },
    { "adcDelay",            adcDelay,            METH_NOARGS, "self.adcDelay() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Ipimb::ConfigV2 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Ipimb::ConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "ConfigV2", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::Ipimb::ConfigV2* obj = pypdsdata::Ipimb::ConfigV2::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "ipimb.ConfigV2(triggerCounter=" << obj->triggerCounter()
      << ", serialID=" << obj->serialID()
      << ", chargeAmpRange=" << obj->chargeAmpRange()
      << ", ...)" ;

  return PyString_FromString( str.str().c_str() );
}

}
