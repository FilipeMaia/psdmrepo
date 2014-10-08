//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_PulseConfigV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PulseConfigV3.h"

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

  // type-specific methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfigV3, pulseId)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfigV3, polarity)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfigV3, prescale)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfigV3, delay)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfigV3, width)

  PyMethodDef methods[] = {
    { "pulseId",            pulseId,            METH_NOARGS, "self.pulseId() -> int\n\nReturns integer number" },
    { "polarity",           polarity,           METH_NOARGS, "self.polarity() -> int\n\nReturns 0 for positive polarity , 1 for negative polarity" },
    { "prescale",           prescale,           METH_NOARGS, "self.prescale() -> int\n\nReturns clock divider" },
    { "delay",              delay,              METH_NOARGS, "self.delay() -> int\n\nReturns delay in 119MHz clks" },
    { "width",              width,              METH_NOARGS, "self.width() -> int\n\nReturns width in 119MHz clks" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::PulseConfigV3 class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::PulseConfigV3::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "PulseConfigV3", module );
}
