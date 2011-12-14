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
#include "ConfigV1.h"
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  PyObject* timestamps( PyObject* self, PyObject* );
  PyObject* channelValues( PyObject* self, PyObject* args );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "timestamps",    timestamps,    METH_NOARGS, "" },
    { "channelValues", channelValues, METH_VARARGS, "" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Gsc16ai::DataV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Gsc16ai::DataV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "DataV1", module );
}

namespace {

PyObject*
channelValues( PyObject* self, PyObject* args )
{
  Pds::Gsc16ai::DataV1* obj = pypdsdata::Gsc16ai::DataV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:Gsc16ai.DataV1.data", &configObj ) ) return 0;

  // size of data array
  npy_intp dims[1] = { 0 };

  // get dimensions from config object
  if ( pypdsdata::Gsc16ai::ConfigV1::Object_TypeCheck( configObj ) ) {
    Pds::Gsc16ai::ConfigV1* config = pypdsdata::Gsc16ai::ConfigV1::pdsObject( configObj );
    dims[0] = config->lastChan() - config->firstChan() + 1;
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a Gsc16ai.ConfigV1 object");
    return 0;
  }

  // NumPy type number
  int typenum = NPY_USHORT ;
  int flags = NPY_C_CONTIGUOUS ;

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 1, dims, typenum, 0,
                                (void*)obj->_channelValue, 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
timestamps( PyObject* self, PyObject* )
{
  Pds::Gsc16ai::DataV1* obj = pypdsdata::Gsc16ai::DataV1::pdsObject( self );
  if ( not obj ) return 0;

  // size of data array
  npy_intp dims[1] = { 3 };

  // NumPy type number
  int typenum = NPY_USHORT ;
  int flags = NPY_C_CONTIGUOUS ;

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 1, dims, typenum, 0,
                                (void*)obj->_timestamp, 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
_repr( PyObject *self )
{
  Pds::Gsc16ai::DataV1* obj = pypdsdata::Gsc16ai::DataV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "Gsc16ai.DataV1(timestamps=[" << obj->_timestamp[0]
      << ", " << obj->_timestamp[1]
      << ", " << obj->_timestamp[2] 
      << "], channelValues=[" << obj->_channelValue[0]
      << ", ...])";

  return PyString_FromString( str.str().c_str() );
}

}
