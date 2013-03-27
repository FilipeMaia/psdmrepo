//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FrameV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "FrameV1.h"

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
  FUN0_WRAPPER(pypdsdata::Fli::FrameV1, shotIdStart)
  FUN0_WRAPPER(pypdsdata::Fli::FrameV1, readoutTime)
  FUN0_WRAPPER(pypdsdata::Fli::FrameV1, temperature)
  PyObject* data( PyObject* self, PyObject* args );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "shotIdStart",    shotIdStart,    METH_NOARGS, "self.shotIdStart() -> int\n\nReturns integer number" },
    { "readoutTime",    readoutTime,    METH_NOARGS, "self.readoutTime() -> float\n\nReturns floating number" },
    { "temperature",    temperature,    METH_NOARGS, "self.temperature() -> float\n\nReturns floating number" },
    { "data",           data,           METH_VARARGS, "self.data(cfg: ConfigV*) -> numpy.ndarray\n\nReturns 2-dim array of integer numbers" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Fli::FrameV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Fli::FrameV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "FrameV1", module );
}

namespace {

PyObject*
data( PyObject* self, PyObject* args )
{
  Pds::Fli::FrameV1* obj = pypdsdata::Fli::FrameV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:Fli.FrameV1.data", &configObj ) ) return 0;

  // dimensions
  npy_intp dims[2] = { 0, 0 };

  // get dimensions from config object
  if ( pypdsdata::Fli::ConfigV1::Object_TypeCheck( configObj ) ) {
    Pds::Fli::ConfigV1* config = pypdsdata::Fli::ConfigV1::pdsObject( configObj );
    uint32_t binX = config->binX();
    uint32_t binY = config->binY();
    dims[0] = (config->height() + binY - 1) / binY;
    dims[1] = (config->width() + binX - 1) / binX;
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a PNCCD.ConfigV* object");
    return 0;
  }

  // NumPy type number
  int typenum = NPY_USHORT ;
  int flags = NPY_C_CONTIGUOUS ;

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 2, dims, typenum, 0,
                                (void*)obj->data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
_repr( PyObject *self )
{
  Pds::Fli::FrameV1* obj = pypdsdata::Fli::FrameV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "Fli.FrameV1(shotIdStart=" << obj->shotIdStart()
      << ", readoutTime=" << obj->readoutTime()
      << ", temperature=" << obj->temperature()
      << ", ...)";

  return PyString_FromString( str.str().c_str() );
}

}
