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
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "DataV1.h"

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
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, triggerCounter)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, config0)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, config1)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, config2)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel0)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel1)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel2)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, channel3)
  FUN0_WRAPPER(pypdsdata::Ipimb::DataV1, checksum)

  PyMethodDef methods[] = {
    { "triggerCounter", triggerCounter, METH_NOARGS, "" },
    { "config0",        config0,        METH_NOARGS, "" },
    { "config1",        config1,        METH_NOARGS, "" },
    { "config2",        config2,        METH_NOARGS, "" },
    { "channel0",       channel0,       METH_NOARGS, "" },
    { "channel1",       channel1,       METH_NOARGS, "" },
    { "channel2",       channel2,       METH_NOARGS, "" },
    { "channel3",       channel3,       METH_NOARGS, "" },
    { "checksum",       checksum,       METH_NOARGS, "" },
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
