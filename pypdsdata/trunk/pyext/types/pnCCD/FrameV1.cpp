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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigV1.h"
#include "ConfigV2.h"
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../camera/FrameCoord.h"
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

  PyMethodDef methods[] = {
    {"specialWord", specialWord, METH_NOARGS,  "" },
    {"frameNumber", frameNumber, METH_NOARGS,  "" },
    {"timeStampHi", timeStampHi, METH_NOARGS,  "" },
    {"timeStampLo", timeStampLo, METH_NOARGS,  "" },
    {"next",        next,        METH_VARARGS, "" },
    {"data",        data,        METH_VARARGS, "" },
    {"sizeofData",  sizeofData,  METH_VARARGS, "" },
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

}
