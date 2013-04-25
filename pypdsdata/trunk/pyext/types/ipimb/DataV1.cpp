//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataV1.h"

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
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, triggerCounter)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, config0)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, config1)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, config2)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel0)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel1)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel2)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel3)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel0Volts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel1Volts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel2Volts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel3Volts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, checksum)

  PyMethodDef methods[] = {
    { "triggerCounter", triggerCounter, METH_NOARGS, "self.triggerCounter() -> int\n\nReturns integer number" },
    { "config0",        config0,        METH_NOARGS, "self.config0() -> int\n\nReturns integer number" },
    { "config1",        config1,        METH_NOARGS, "self.config1() -> int\n\nReturns integer number" },
    { "config2",        config2,        METH_NOARGS, "self.config2() -> int\n\nReturns integer number" },
    { "channel0",       channel0,       METH_NOARGS, "self.channel0() -> int\n\nReturns integer number" },
    { "channel1",       channel1,       METH_NOARGS, "self.channel1() -> int\n\nReturns integer number" },
    { "channel2",       channel2,       METH_NOARGS, "self.channel2() -> int\n\nReturns integer number" },
    { "channel3",       channel3,       METH_NOARGS, "self.channel3() -> int\n\nReturns integer number" },
    { "channel0Volts",  channel0Volts,  METH_NOARGS, "self.channel0Volts() -> float\n\nReturns floating number" },
    { "channel1Volts",  channel1Volts,  METH_NOARGS, "self.channel1Volts() -> float\n\nReturns floating number" },
    { "channel2Volts",  channel2Volts,  METH_NOARGS, "self.channel2Volts() -> float\n\nReturns floating number" },
    { "channel3Volts",  channel3Volts,  METH_NOARGS, "self.channel3Volts() -> float\n\nReturns floating number" },
    { "checksum",       checksum,       METH_NOARGS, "self.checksum() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Ipimb::DataV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Ipimb::DataV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "DataV1", module );
}

void
pypdsdata::Ipimb::DataV1::print(std::ostream& str) const
{
  str << "ipimb.DataV1(triggerCounter=" << m_obj->triggerCounter()
      << ", v0=" << m_obj->channel0Volts()
      << ", v1=" << m_obj->channel1Volts()
      << ", v2=" << m_obj->channel2Volts()
      << ", v3=" << m_obj->channel3Volts()
      << ", ...)" ;
}
