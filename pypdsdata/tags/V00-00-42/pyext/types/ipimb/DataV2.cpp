//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataV2.h"

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
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, triggerCounter)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, config0)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, config1)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, config2)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel0)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel1)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel2)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel3)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel0ps)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel1ps)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel2ps)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel3ps)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel0Volts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel1Volts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel2Volts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel3Volts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel0psVolts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel1psVolts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel2psVolts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, channel3psVolts)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV2, checksum)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "triggerCounter", triggerCounter, METH_NOARGS, "" },
    { "config0",        config0,        METH_NOARGS, "" },
    { "config1",        config1,        METH_NOARGS, "" },
    { "config2",        config2,        METH_NOARGS, "" },
    { "channel0",       channel0,       METH_NOARGS, "" },
    { "channel1",       channel1,       METH_NOARGS, "" },
    { "channel2",       channel2,       METH_NOARGS, "" },
    { "channel3",       channel3,       METH_NOARGS, "" },
    { "channel0ps",     channel0ps,     METH_NOARGS, "" },
    { "channel1ps",     channel1ps,     METH_NOARGS, "" },
    { "channel2ps",     channel2ps,     METH_NOARGS, "" },
    { "channel3ps",     channel3ps,     METH_NOARGS, "" },
    { "channel0Volts",  channel0Volts,  METH_NOARGS, "" },
    { "channel1Volts",  channel1Volts,  METH_NOARGS, "" },
    { "channel2Volts",  channel2Volts,  METH_NOARGS, "" },
    { "channel3Volts",  channel3Volts,  METH_NOARGS, "" },
    { "channel0psVolts", channel0psVolts, METH_NOARGS, "" },
    { "channel1psVolts", channel1psVolts, METH_NOARGS, "" },
    { "channel2psVolts", channel2psVolts, METH_NOARGS, "" },
    { "channel3psVolts", channel3psVolts, METH_NOARGS, "" },
    { "checksum",       checksum,       METH_NOARGS, "" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Ipimb::DataV2 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Ipimb::DataV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "DataV2", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::Ipimb::DataV2* obj = pypdsdata::Ipimb::DataV2::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "ipimb.DataV2(triggerCounter=" << obj->triggerCounter()
      << ", v0=" << obj->channel0Volts()
      << ", v1=" << obj->channel1Volts()
      << ", v2=" << obj->channel2Volts()
      << ", v3=" << obj->channel3Volts()
      << ", ...)" ;

  return PyString_FromString( str.str().c_str() );
}

}
