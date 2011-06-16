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
    { "triggerCounter",      triggerCounter,      METH_NOARGS, "" },
    { "serialID",            serialID,            METH_NOARGS, "" },
    { "chargeAmpRange",      chargeAmpRange,      METH_NOARGS, "" },
    { "calibrationRange",    calibrationRange,    METH_NOARGS, "" },
    { "resetLength",         resetLength,         METH_NOARGS, "" },
    { "resetDelay",          resetDelay,          METH_NOARGS, "" },
    { "chargeAmpRefVoltage", chargeAmpRefVoltage, METH_NOARGS, "" },
    { "calibrationVoltage",  calibrationVoltage,  METH_NOARGS, "" },
    { "diodeBias",           diodeBias,           METH_NOARGS, "" },
    { "status",              status,              METH_NOARGS, "" },
    { "errors",              errors,              METH_NOARGS, "" },
    { "calStrobeLength",     calStrobeLength,     METH_NOARGS, "" },
    { "trigDelay",           trigDelay,           METH_NOARGS, "" },
    { "trigPsDelay",         trigPsDelay,         METH_NOARGS, "" },
    { "adcDelay",            adcDelay,            METH_NOARGS, "" },
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
