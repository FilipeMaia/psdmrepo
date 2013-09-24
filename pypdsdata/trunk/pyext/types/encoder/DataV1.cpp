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
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // methods
  namespace int_getset {
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Encoder::DataV1, timestamp)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Encoder::DataV1, encoder_count)
  }

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"_33mhz_timestamp", int_getset::timestamp,  0, "Integer number", 0},
    {"_encoder_count",   int_getset::encoder_count,    0, "Integer number", 0},
    {0, 0, 0, 0, 0}
  };

  // methods
  namespace int_methods {
  FUN0_WRAPPER(pypdsdata::Encoder::DataV1, value)
  FUN0_WRAPPER(pypdsdata::Encoder::DataV1, timestamp)
  FUN0_WRAPPER(pypdsdata::Encoder::DataV1, encoder_count)
  }

  PyMethodDef methods[] = {
    {"timestamp",      int_methods::timestamp,      METH_NOARGS,  "self.timestamp() -> int\n\nReturns integer number" },
    {"encoder_count",  int_methods::encoder_count,  METH_NOARGS,  "self.encoder_count() -> int\n\nReturns integer number" },
    {"value",          int_methods::value,          METH_NOARGS,  "self.value() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Encoder::DataV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Encoder::DataV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_methods = ::methods;

  BaseType::initType( "DataV1", module );
}

void
pypdsdata::Encoder::DataV1::print(std::ostream& str) const
{
  str << "encoder.DataV1(33mhz_timestamp=" << m_obj->timestamp()
      << ", encoder_count=" << m_obj->encoder_count()
      << ")";
}
