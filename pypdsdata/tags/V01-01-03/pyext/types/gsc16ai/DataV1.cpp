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
  PyObject* timestamp( PyObject* self, PyObject* );
  PyObject* channelValues( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "timestamp",     timestamp,     METH_NOARGS,  "self.timestamp() -> ndarray\n\nReturns array of integers" },
    { "channelValues", channelValues, METH_VARARGS, "self.channelValues(cfg: ConfigV1) -> ndarray\n\nReturns array of integers" },
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

  BaseType::initType( "DataV1", module );
}

void
pypdsdata::Gsc16ai::DataV1::print(std::ostream& str) const
{
  str << "Gsc16ai.DataV1(timestamp=" << m_obj->timestamp()
      << ", ...)";
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
  Pds::Gsc16ai::ConfigV1* config = 0;
  if ( pypdsdata::Gsc16ai::ConfigV1::Object_TypeCheck( configObj ) ) {
    config = pypdsdata::Gsc16ai::ConfigV1::pdsObject( configObj );
    dims[0] = config->numChannels();
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a Gsc16ai.ConfigV1 object");
    return 0;
  }

  // NumPy type number
  int typenum = NPY_USHORT ;
  int flags = NPY_C_CONTIGUOUS ;

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 1, dims, typenum, 0,
                                (void*)obj->channelValue(*config).data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
timestamp( PyObject* self, PyObject* )
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
                                (void*)obj->timestamp().data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

}
