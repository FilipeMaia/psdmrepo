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

  PyObject* next( PyObject* self, PyObject* args );
  PyObject* data( PyObject* self, PyObject* args );
  PyObject* sizeofData( PyObject* self, PyObject* args );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"specialWord", specialWord, METH_NOARGS,  "self.specialWord() -> int\n\nReturns integer number" },
    {"frameNumber", frameNumber, METH_NOARGS,  "self.frameNumber() -> int\n\nReturns integer number" },
    {"timeStampHi", timeStampHi, METH_NOARGS,  "self.timeStampHi() -> int\n\nReturns integer number" },
    {"timeStampLo", timeStampLo, METH_NOARGS,  "self.timeStampLo() -> int\n\nReturns integer number" },
    {"next",        next,        METH_VARARGS,
        "self.next(cfg: ConfigV*) -> FrameV1\n\nReturns frame object (:py:class:`FrameV1`) for the next link, takes config object as argument" },
    {"data",        data,        METH_VARARGS, 
        "self.data(cfg: ConfigV*) -> numpy.ndarray\n\nReturns frame data as NumPy 2-dimensional array of integers of size 512x512" },
    {"sizeofData",  sizeofData,  METH_VARARGS, "self.sizeofData(cfg: ConfigV*) -> int\n\nReturns size of data in a frame (in pixels)" },
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "FrameV1", module );
}

namespace {

PyObject*
next( PyObject* self, PyObject* args )
{
  Pds::PNCCD::FrameV1* obj = pypdsdata::PNCCD::FrameV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:PNCCD.FrameV1.next", &configObj ) ) return 0;

  // get Pds::PNCCD::ConfigV1 from argument which could also be of Config2 type
  uint32_t payloadSizePerLink = 0;
  Pds::PNCCD::FrameV1* next = 0;
  if ( pypdsdata::PNCCD::ConfigV1::Object_TypeCheck( configObj ) ) {
    const Pds::PNCCD::ConfigV1* config = pypdsdata::PNCCD::ConfigV1::pdsObject( configObj );
    next = const_cast<Pds::PNCCD::FrameV1*>(obj->next( *config ));
    payloadSizePerLink = config->payloadSizePerLink();
  } else {
    if ( pypdsdata::PNCCD::ConfigV2::Object_TypeCheck( configObj ) ) {
      Pds::PNCCD::ConfigV2* config = pypdsdata::PNCCD::ConfigV2::pdsObject( configObj );
      next = const_cast<Pds::PNCCD::FrameV1*>(obj->next( *config ));
      payloadSizePerLink = config->payloadSizePerLink();
    } else {
      PyErr_SetString(PyExc_TypeError, "Error: parameter is not a PNCCD.ConfigV1 object");
      return 0;
    }
  }

  // make Python object
  pypdsdata::PNCCD::FrameV1* py_this = (pypdsdata::PNCCD::FrameV1*) self;
  return pypdsdata::PNCCD::FrameV1::PyObject_FromPds( next, py_this->m_parent, payloadSizePerLink, py_this->m_dtor );
}

PyObject*
data( PyObject* self, PyObject* args )
{
  const Pds::PNCCD::FrameV1* obj = pypdsdata::PNCCD::FrameV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:PNCCD.FrameV1.data", &configObj ) ) return 0;

  // get Pds::PNCCD::ConfigV1 from argument which could also be of Config2 type
  unsigned size = 0;
  if ( pypdsdata::PNCCD::ConfigV1::Object_TypeCheck( configObj ) ) {
    const Pds::PNCCD::ConfigV1* config = pypdsdata::PNCCD::ConfigV1::pdsObject( configObj );
    size = obj->sizeofData( *config );
  } else {
    if ( pypdsdata::PNCCD::ConfigV2::Object_TypeCheck( configObj ) ) {
      const Pds::PNCCD::ConfigV2* config = pypdsdata::PNCCD::ConfigV2::pdsObject( configObj );
      size = obj->sizeofData( *config );
    } else {
      PyErr_SetString(PyExc_TypeError, "Error: parameter is not a PNCCD.ConfigV1 object");
      return 0;
    }
  }

  // asume that single frame is 512x512 image
  if ( size != 512*512 ) {
    PyErr_Format(pypdsdata::exceptionType(), "Error: odd size of frame data, expect 512x512, received %d", size);
    return 0;
  }

  // NumPy type number
  int typenum = NPY_USHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  npy_intp dims[2] = { 512, 512 };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 2, dims, typenum, 0,
                                (void*)obj->data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
sizeofData( PyObject* self, PyObject* args )
{
  const Pds::PNCCD::FrameV1* obj = pypdsdata::PNCCD::FrameV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:PNCCD.FrameV1.sizeofData", &configObj ) ) return 0;

  // get Pds::PNCCD::ConfigV1 from argument which could also be of Config2 type
  unsigned size = 0;
  if ( pypdsdata::PNCCD::ConfigV1::Object_TypeCheck( configObj ) ) {
    const Pds::PNCCD::ConfigV1* config = pypdsdata::PNCCD::ConfigV1::pdsObject( configObj );
    size = obj->sizeofData( *config );
  } else {
    if ( pypdsdata::PNCCD::ConfigV2::Object_TypeCheck( configObj ) ) {
      const Pds::PNCCD::ConfigV2* config = pypdsdata::PNCCD::ConfigV2::pdsObject( configObj );
      size = obj->sizeofData( *config );
    } else {
      PyErr_SetString(PyExc_TypeError, "Error: parameter is not a PNCCD.ConfigV1 object");
      return 0;
    }
  }

  return PyInt_FromLong( size );
}

PyObject*
_repr( PyObject *self )
{
  Pds::PNCCD::FrameV1* obj = pypdsdata::PNCCD::FrameV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "pnccd.FrameV1(specialWord=" << obj->specialWord()
      << ", frameNumber=" << obj->frameNumber()
      << ", timeStampHi=" << obj->timeStampHi()
      << ", timeStampLo=" << obj->timeStampLo()
      << ", ...)";

  return PyString_FromString( str.str().c_str() );
}

}
