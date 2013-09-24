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
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::Camera::FrameV1, width)
  FUN0_WRAPPER(pypdsdata::Camera::FrameV1, height)
  FUN0_WRAPPER(pypdsdata::Camera::FrameV1, depth)
//  FUN0_WRAPPER(pypdsdata::Camera::FrameV1, depth_bytes)
  FUN0_WRAPPER(pypdsdata::Camera::FrameV1, offset)
  PyObject* FrameV1_data( PyObject* self, PyObject* );

  PyMethodDef methods[] = {
    {"width",       width,         METH_NOARGS,  "self.width() -> int\n\nReturns number of pixels in a row." },
    {"height",      height,        METH_NOARGS,  "self.height() -> int\n\nReturns number of pixels in a column." },
    {"depth",       depth,         METH_NOARGS,  "self.depth() -> int\n\nReturns number of bits per pixel." },
//    {"depth_bytes", depth_bytes,   METH_NOARGS,  "self.depth_bytes() -> int\n\nReturns number of bytes per pixel." },
    {"offset",      offset,        METH_NOARGS,  "self.offset() -> int\n\nReturns fixed offset/pedestal value of pixel data." },
    {"data",        FrameV1_data,  METH_VARARGS,
        "self.data(writable = False) -> numpy.ndarray\n\nReturns pixel data as NumPy array, if optional argument is True then array is writable." },
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

  BaseType::initType( "FrameV1", module );
}

void
pypdsdata::Camera::FrameV1::print(std::ostream& out) const
{
  out << "camera.FrameV1(" << m_obj->width() << "x" << m_obj->height() << "x" << m_obj->depth() << ")";
}

namespace {

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
  if ( obj->depth() <= 8 ) {
    typenum = NPY_UBYTE;
  } else if ( obj->depth() <= 16 ) {
    typenum = NPY_USHORT;
  } else {
    typenum = NPY_UINT;
  }

  // allow it to be writable on user's request
  int flags = NPY_C_CONTIGUOUS ;
  if ( writable ) flags |= NPY_WRITEABLE;

  // dimensions
  npy_intp dims[2] = { obj->height(), obj->width() };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 2, dims, typenum, 0,
                                (void*)obj->_int_pixel_data().data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

}
