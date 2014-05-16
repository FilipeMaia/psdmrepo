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
  FUN0_WRAPPER(pypdsdata::Pimax::FrameV1, shotIdStart)
  FUN0_WRAPPER(pypdsdata::Pimax::FrameV1, readoutTime)
  FUN0_WRAPPER(pypdsdata::Pimax::FrameV1, temperature)
  PyObject* data( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "shotIdStart",    shotIdStart,    METH_NOARGS, "self.shotIdStart() -> int\n\nReturns integer number" },
    { "readoutTime",    readoutTime,    METH_NOARGS, "self.readoutTime() -> float\n\nReturns floating number" },
    { "temperature",    temperature,    METH_NOARGS, "self.temperature() -> float\n\nReturns floating number" },
    { "data",           data,           METH_VARARGS, "self.data(cfg: ConfigV*) -> numpy.ndarray\n\nReturns 2-dim array of integer numbers" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Pimax::FrameV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Pimax::FrameV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "FrameV1", module );
}

void
pypdsdata::Pimax::FrameV1::print(std::ostream& str) const
{
  str << "Pimax.FrameV1(shotIdStart=" << m_obj->shotIdStart()
      << ", readoutTime=" << m_obj->readoutTime()
      << ", temperature=" << m_obj->temperature()
      << ", ...)";
}

namespace {

PyObject*
data( PyObject* self, PyObject* args )
{
  Pds::Pimax::FrameV1* obj = pypdsdata::Pimax::FrameV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:Pimax.FrameV1.data", &configObj ) ) return 0;

  // dimensions
  npy_intp dims[2] = { 0, 0 };

  // get dimensions from config object
  Pds::Pimax::ConfigV1* config = 0;
  if ( pypdsdata::Pimax::ConfigV1::Object_TypeCheck( configObj ) ) {
    config = pypdsdata::Pimax::ConfigV1::pdsObject( configObj );
    dims[0] = config->numPixelsY();
    dims[1] = config->numPixelsX();
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a Pimax.ConfigV* object");
    return 0;
  }

  // NumPy type number
  int typenum = NPY_USHORT ;
  int flags = NPY_C_CONTIGUOUS ;

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 2, dims, typenum, 0,
                                (void*)obj->data(*config).data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

}
