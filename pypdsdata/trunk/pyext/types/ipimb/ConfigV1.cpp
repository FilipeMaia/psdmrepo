//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1.h"

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

  // Ipimb::ConfigV1 does not defie this enum, this come from communication
  // with Matt:
  //  IpimbConfigV1
  //  
  //     { 1pF, 100pF, 10nF }
  //  
  //     gain of channel0 = (chargeAmpRange()>>0)&0x3
  //     gain of channel1 = (chargeAmpRange()>>2)&0x3
  //     ..
  //     (only 8 bits are used)
  pypdsdata::EnumType::Enum capacitorValueEnumValues[] = {
      { "c_1pF",       0 },
      { "c_100pF",     1 },
      { "c_10nF",      2 },
      { 0, 0 }
  };
  pypdsdata::EnumType capacitorValueEnum ( "CapacitorValue", capacitorValueEnumValues );

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, triggerCounter)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, serialID)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, chargeAmpRange)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, calibrationRange)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, resetLength)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, resetDelay)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, chargeAmpRefVoltage)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, calibrationVoltage)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, diodeBias)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, status)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, errors)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, calStrobeLength)
  FUN0_WRAPPER(pypdsdata::Ipimb::ConfigV1, trigDelay)
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
    { "diodeGain",           diodeGain,           METH_VARARGS, 
        "self.diodeGain(ch: int) -> CapacitorValue enum\n\nReturns CapacitorValue enum for given channel number (0..3)" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Ipimb::ConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Ipimb::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "CapacitorValue", capacitorValueEnum.type() );

  BaseType::initType( "ConfigV1", module );
}

namespace {
  
PyObject*
diodeGain( PyObject* self, PyObject* args )
{
  const Pds::Ipimb::ConfigV1* obj = pypdsdata::Ipimb::ConfigV1::pdsObject(self);
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:Ipimb_ConfigV1_diodeGain", &index ) ) return 0;

  if ( index >= 4 ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..3] in Ipimb.ConfigV1.diodeGain()");
    return 0;
  }
  
  return capacitorValueEnum.Enum_FromLong((obj->chargeAmpRange() >> (2*index)) & 0x3);
}

PyObject*
_repr( PyObject *self )
{
  const Pds::Ipimb::ConfigV1* obj = pypdsdata::Ipimb::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "ipimb.ConfigV1(triggerCounter=" << obj->triggerCounter()
      << ", serialID=" << obj->serialID()
      << ", chargeAmpRange=" << obj->chargeAmpRange()
      << ", ...)" ;

  return PyString_FromString( str.str().c_str() );
}

}
