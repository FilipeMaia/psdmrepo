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
  MEMBER_WRAPPER(pypdsdata::Encoder::DataV2, _33mhz_timestamp)
  PyObject* _encoder_count( PyObject* self, void* );
  PyObject* _repr( PyObject *self );

  PyGetSetDef getset[] = {
    {"_33mhz_timestamp", _33mhz_timestamp,  0, "Integer number", 0},
    {"_encoder_count",   _encoder_count,    0, "List of 3 integer numbers", 0},
    {0, 0, 0, 0, 0}
  };

  // methods
  PyObject* value( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"value",     value,       METH_VARARGS,  "self.value(chan: int) -> int\n\nReturns value for given channel number (0..2)" },
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "DataV2", module );
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


PyObject*
_encoder_count( PyObject* self, void* )
{
  const Pds::Encoder::DataV2* obj = pypdsdata::Encoder::DataV2::pdsObject( self );
  if ( not obj ) return 0;

  const int size = 3;
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    PyList_SET_ITEM( list, i, pypdsdata::TypeLib::toPython(obj->_encoder_count[i]) );
  }
  return list;
}


PyObject*
_repr( PyObject *self )
{
  Pds::Encoder::DataV2* obj = pypdsdata::Encoder::DataV2::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "encoder.DataV1(33mhz_timestamp=" << obj->_33mhz_timestamp
      << ", encoder_count=[" << obj->_encoder_count[0]
      << ", " << obj->_encoder_count[1]
      << ", " << obj->_encoder_count[2]
      << "])";
  
  return PyString_FromString( str.str().c_str() );
}

}
