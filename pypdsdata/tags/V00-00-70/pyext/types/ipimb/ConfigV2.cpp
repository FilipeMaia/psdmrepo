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
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  pypdsdata::EnumType::Enum capacitorValueEnumValues[] = {
      { "c_1pF",         Pds::Ipimb::ConfigV2::c_1pF },
      { "c_4p7pF",       Pds::Ipimb::ConfigV2::c_4p7pF },
      { "c_24pF",        Pds::Ipimb::ConfigV2::c_24pF },
      { "c_120pF",       Pds::Ipimb::ConfigV2::c_120pF },
      { "c_620pF",       Pds::Ipimb::ConfigV2::c_620pF },
      { "c_3p3nF",       Pds::Ipimb::ConfigV2::c_3p3nF },
      { "c_10nF",        Pds::Ipimb::ConfigV2::c_10nF },
      { "expert",        Pds::Ipimb::ConfigV2::expert },
      { 0, 0 }
  };
  pypdsdata::EnumType capacitorValueEnum ( "CapacitorValue", capacitorValueEnumValues );

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
  PyObject* diodeGain(PyObject* self, PyObject* args);
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
    { "diodeGain",           diodeGain,           METH_VARARGS, 
        "self.diodeGain(ch: int) -> CapacitorValue enum\n\nReturns :py:class:`CapacitorValue` enum for given channel number (0..3)" },
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

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "CapacitorValue", capacitorValueEnum.type() );

  BaseType::initType( "ConfigV2", module );
}

namespace {

PyObject*
diodeGain( PyObject* self, PyObject* args )
{
  const Pds::Ipimb::ConfigV2* obj = pypdsdata::Ipimb::ConfigV2::pdsObject(self);
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:Ipimb_ConfigV2_diodeGain", &index ) ) return 0;

  if ( index >= 4 ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..3] in Ipimb.ConfigV2.diodeGain()");
    return 0;
  }
  
  return capacitorValueEnum.Enum_FromLong((obj->chargeAmpRange() >> (4*index)) & 0xf);
}

PyObject*
_repr( PyObject *self )
{
  const Pds::Ipimb::ConfigV2* obj = pypdsdata::Ipimb::ConfigV2::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "ipimb.ConfigV2(triggerCounter=" << obj->triggerCounter()
      << ", serialID=" << obj->serialID()
      << ", chargeAmpRange=" << obj->chargeAmpRange()
      << ", ...)" ;

  return PyString_FromString( str.str().c_str() );
}

}
