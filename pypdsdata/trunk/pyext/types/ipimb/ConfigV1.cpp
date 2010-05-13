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
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

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

  BaseType::initType( "ConfigV1", module );
}
