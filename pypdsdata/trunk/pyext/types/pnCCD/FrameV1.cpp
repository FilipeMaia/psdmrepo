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

  // methods
  FUN0_WRAPPER(pypdsdata::PNCCD::FrameV1, specialWord)
  FUN0_WRAPPER(pypdsdata::PNCCD::FrameV1, frameNumber)
  FUN0_WRAPPER(pypdsdata::PNCCD::FrameV1, timeStampHi)
  FUN0_WRAPPER(pypdsdata::PNCCD::FrameV1, timeStampLo)
  PyObject* data( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"specialWord", specialWord, METH_NOARGS,  "self.specialWord() -> int\n\nReturns integer number" },
    {"frameNumber", frameNumber, METH_NOARGS,  "self.frameNumber() -> int\n\nReturns integer number" },
    {"timeStampHi", timeStampHi, METH_NOARGS,  "self.timeStampHi() -> int\n\nReturns integer number" },
    {"timeStampLo", timeStampLo, METH_NOARGS,  "self.timeStampLo() -> int\n\nReturns integer number" },
    {"data",        data,        METH_VARARGS, 
        "self.data(cfg: ConfigV*) -> numpy.ndarray\n\nReturns frame data as NumPy 2-dimensional array of integers of size 512x512" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::PNCCD::FrameV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::PNCCD::FrameV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "FrameV1", module );
}

void
pypdsdata::PNCCD::FrameV1::print(std::ostream& str) const
{
  str << "pnccd.FrameV1(specialWord=" << m_obj->specialWord()
      << ", frameNumber=" << m_obj->frameNumber()
      << ", timeStampHi=" << m_obj->timeStampHi()
      << ", timeStampLo=" << m_obj->timeStampLo()
      << " ...)";
}

namespace {

PyObject*
data( PyObject* self, PyObject* args )
{
  const Pds::PNCCD::FrameV1* obj = pypdsdata::PNCCD::FrameV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:PNCCD.FrameV1.data", &configObj ) ) return 0;

  // get Pds::PNCCD::ConfigV1 from argument which could also be of Config2 type
  ndarray<const uint16_t, 2> data;;
  if ( pypdsdata::PNCCD::ConfigV1::Object_TypeCheck( configObj ) ) {
    Pds::PNCCD::ConfigV1* config = pypdsdata::PNCCD::ConfigV1::pdsObject( configObj );
    data = obj->data(*config);
  } else if ( pypdsdata::PNCCD::ConfigV2::Object_TypeCheck( configObj ) ) {
    Pds::PNCCD::ConfigV2* config = pypdsdata::PNCCD::ConfigV2::pdsObject( configObj );
    data = obj->data(*config);
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a PNCCD.ConfigV* object");
    return 0;
  }

  // assume that single frame is 512x512 image
  const unsigned* shape = data.shape();
  if ( shape[0] != 512 or shape[1] != 512 ) {
    PyErr_Format(pypdsdata::exceptionType(), "Error: odd size of frame data, expect 512x512, received %dx%d", shape[0], shape[1]);
    return 0;
  }

  // NumPy type number
  int typenum = NPY_USHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  npy_intp dims[2] = { shape[0], shape[1] };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 2, dims, typenum, 0,
                                (void*)data.data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

}
