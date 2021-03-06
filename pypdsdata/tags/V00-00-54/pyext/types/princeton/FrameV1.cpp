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
#include "ConfigV2.h"
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Princeton::FrameV1, shotIdStart)
  FUN0_WRAPPER(pypdsdata::Princeton::FrameV1, readoutTime)
  PyObject* data( PyObject* self, PyObject* args );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "shotIdStart",    shotIdStart,    METH_NOARGS, "self.shotIdStart() -> int\n\nReturns integer number" },
    { "readoutTime",    readoutTime,    METH_NOARGS, "self.readoutTime() -> float\n\nReturns floating number" },
    { "data",           data,           METH_VARARGS, "self.data(cfg: ConfigV*) -> numpy.ndarray\n\nReturns 2-dim array of integer numbers" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Princeton::FrameV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Princeton::FrameV1::initType( PyObject* module )
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
  Pds::Princeton::FrameV1* obj = pypdsdata::Princeton::FrameV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:Princeton.FrameV1.data", &configObj ) ) return 0;

  // dimensions
  npy_intp dims[2] = { 0, 0 };

  // get dimensions from config object
  if ( pypdsdata::Princeton::ConfigV1::Object_TypeCheck( configObj ) ) {
    Pds::Princeton::ConfigV1* config = pypdsdata::Princeton::ConfigV1::pdsObject( configObj );
    dims[0] = config->height();
    dims[1] = config->width();
  } else if ( pypdsdata::Princeton::ConfigV2::Object_TypeCheck( configObj ) ) {
    Pds::Princeton::ConfigV2* config = pypdsdata::Princeton::ConfigV2::pdsObject( configObj );
    dims[0] = config->height();
    dims[1] = config->width();
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
  Pds::Princeton::FrameV1* obj = pypdsdata::Princeton::FrameV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "princeton.FrameV1(shotIdStart=" << obj->shotIdStart()
      << ", readoutTime=" << obj->readoutTime()
      << ", ...)";

  return PyString_FromString( str.str().c_str() );
}

}
