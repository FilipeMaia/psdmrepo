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
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // methods
  namespace int_getset {
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Encoder::DataV2, timestamp)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Encoder::DataV2, encoder_count)
  }

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"_33mhz_timestamp", int_getset::timestamp,     0, "Integer number", 0},
    {"_encoder_count",   int_getset::encoder_count, 0, "List of 3 integer numbers", 0},
    {0, 0, 0, 0, 0}
  };

  // methods
  namespace int_methods {
  FUN0_WRAPPER(pypdsdata::Encoder::DataV2, timestamp)
  FUN0_WRAPPER(pypdsdata::Encoder::DataV2, encoder_count)
  }
  PyObject* value( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"timestamp",      int_methods::timestamp,      METH_NOARGS,  "self.timestamp() -> int\n\nReturns integer number" },
    {"encoder_count",  int_methods::encoder_count,  METH_NOARGS,  "self.encoder_count() -> int\n\nReturns list of integer numbers" },
    {"value",          value,                       METH_VARARGS, "self.value(chan: int) -> int\n\nReturns value for given channel number (0..2)" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Encoder::DataV2 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Encoder::DataV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_methods = ::methods;

  BaseType::initType( "DataV2", module );
}

void
pypdsdata::Encoder::DataV2::print(std::ostream& str) const
{
  str << "encoder.DataV1(33mhz_timestamp=" << m_obj->timestamp()
      << ", encoder_count=" << m_obj->encoder_count()
      << ")";
}

namespace {

PyObject*
value( PyObject* self, PyObject* args )
{
  const Pds::Encoder::DataV2* obj = pypdsdata::Encoder::DataV2::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned index ;
  if ( not PyArg_ParseTuple( args, "I:EncoderDataV2_value", &index ) ) return 0;

  if ( index >= 3 ) {
    PyErr_SetString(PyExc_IndexError, "index outside of range [0..3) in EncoderDataV2.value()");
    return 0;
  }
  
  return PyInt_FromLong( obj->value(index) );
}

}
