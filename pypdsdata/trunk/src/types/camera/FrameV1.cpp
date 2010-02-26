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
#include "Exception.h"
#include "types/TypeLib.h"
#include "pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  PyObject* FrameV1_str( PyObject* self );
  PyObject* FrameV1_repr( PyObject* self );

  // methods
  FUN0_WRAPPER(pypdsdata::Camera::FrameV1, width)
  FUN0_WRAPPER(pypdsdata::Camera::FrameV1, height)
  FUN0_WRAPPER(pypdsdata::Camera::FrameV1, depth)
  FUN0_WRAPPER(pypdsdata::Camera::FrameV1, depth_bytes)
  FUN0_WRAPPER(pypdsdata::Camera::FrameV1, offset)
  FUN0_WRAPPER(pypdsdata::Camera::FrameV1, data_size)
  PyObject* FrameV1_data( PyObject* self, PyObject* );
  PyObject* FrameV1_pixel( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"width",       width,         METH_NOARGS,  "Returns number of pixels in a row." },
    {"height",      height,        METH_NOARGS,  "Returns number of pixels in a column." },
    {"depth",       depth,         METH_NOARGS,  "Returns number of bits per pixel." },
    {"depth_bytes", depth_bytes,   METH_NOARGS,  "Returns number of bytes per pixel." },
    {"offset",      offset,        METH_NOARGS,  "Returns fixed offset/pedestal value of pixel data." },
    {"data_size",   data_size,     METH_NOARGS,  "Returns size of pixel data." },
    {"data",        FrameV1_data,  METH_VARARGS, "Returns pixel data as NumPy array, if optional argument is True then array is writable." },
    {"pixel",       FrameV1_pixel, METH_VARARGS, "Returns individual pixel datum given coordinates (x, y)." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Camera::FrameV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Camera::FrameV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = FrameV1_str;
  type->tp_repr = FrameV1_repr;

  BaseType::initType( "FrameV1", module );
}

namespace {

PyObject*
FrameV1_str( PyObject* self )
{
  const Pds::Camera::FrameV1* obj = pypdsdata::Camera::FrameV1::pdsObject( self );
  if ( not obj ) return 0;

  char buf[64];
  snprintf(buf, sizeof buf, "Camera.FrameV1(%dx%dx%d)",
      obj->width(), obj->height(), obj->depth() );

  return PyString_FromString(buf);
}

PyObject*
FrameV1_repr( PyObject* self )
{
  const Pds::Camera::FrameV1* obj = pypdsdata::Camera::FrameV1::pdsObject( self );
  if ( not obj ) return 0;

  char buf[64];
  snprintf(buf, sizeof buf, "<Camera.FrameV1(%dx%dx%d) at %p>",
      obj->width(), obj->height(), obj->depth(), (void*)self );

  return PyString_FromString(buf);
}

PyObject*
FrameV1_data( PyObject* self, PyObject* args)
{
  const Pds::Camera::FrameV1* obj = pypdsdata::Camera::FrameV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  int writable = 0 ;
  if ( not PyArg_ParseTuple( args, "|i:FrameV1_data", &writable ) ) return 0;

  // NumPy type number
  int typenum = 0 ;
  switch ( obj->depth_bytes() ) {
  case 1:
    typenum = NPY_UBYTE;
    break;
  case 2:
    typenum = NPY_USHORT;
    break;
  case 4:
    typenum = NPY_UINT;
    break;
  }

  // allow it to be writable on user's request
  int flags = NPY_C_CONTIGUOUS ;
  if ( writable ) flags |= NPY_WRITEABLE;

  // dimensions
  npy_intp dims[2] = { obj->height(), obj->width() };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 2, dims, typenum, 0,
                                (void*)obj->data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
FrameV1_pixel( PyObject* self, PyObject* args )
{
  const Pds::Camera::FrameV1* obj = pypdsdata::Camera::FrameV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  unsigned x, y ;
  if ( not PyArg_ParseTuple( args, "II:FrameV1_pixel", &x, &y ) ) return 0;

  const void* data = obj->pixel(x, y);
  switch ( obj->depth_bytes() ) {
  case 1:
    return PyInt_FromLong(*(uint8_t*)data);
  case 2:
    return PyInt_FromLong(*(uint16_t*)data);
  case 4:
    return PyInt_FromLong(*(uint32_t*)data);
  }

  PyErr_SetString(PyExc_TypeError, "Unexpected pixel depth");
  return 0;
}

}
