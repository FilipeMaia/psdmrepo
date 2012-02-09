//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Timepix_TM6740DataV1...
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
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::Timepix::DataV1, width)
  FUN0_WRAPPER(pypdsdata::Timepix::DataV1, height)
  FUN0_WRAPPER(pypdsdata::Timepix::DataV1, depth)
  FUN0_WRAPPER(pypdsdata::Timepix::DataV1, depth_bytes)
  FUN0_WRAPPER(pypdsdata::Timepix::DataV1, data_size)
  FUN0_WRAPPER(pypdsdata::Timepix::DataV1, timestamp)
  FUN0_WRAPPER(pypdsdata::Timepix::DataV1, frameCounter)
  FUN0_WRAPPER(pypdsdata::Timepix::DataV1, lostRows)
  PyObject* data( PyObject* self, PyObject* );
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "width",          width,          METH_NOARGS,  "self.width() -> int\n\nReturns image width" },
    { "height",         height,         METH_NOARGS,  "self.height() -> int\n\nReturns image height" },
    { "depth",          depth,          METH_NOARGS,  "self.depth() -> int\n\nReturns number of bits per pixel" },
    { "depth_bytes",    depth_bytes,    METH_NOARGS,  "self.depth_bytes() -> int\n\nReturns number of bytes per pixel" },
    { "data_size",      data_size,      METH_NOARGS,  "self.data_size() -> int\n\nReturns size of image data" },
    { "timestamp",      timestamp,      METH_NOARGS,  "self.timestamp() -> int\n\nReturns integer number" },
    { "frameCounter",   frameCounter,   METH_NOARGS,  "self.frameCounter() -> int\n\nReturns integer number" },
    { "lostRows",       lostRows,       METH_NOARGS,  "self.lostRows() -> int\n\nReturns integer number" },
    { "data",           data,           METH_NOARGS,  "self.data() -> numpy.ndarray\n\nReturns 2-dim array of integers" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Timepix::DataV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Timepix::DataV1::initType( PyObject* module )
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
data( PyObject* self, PyObject* args )
{
  Pds::Timepix::DataV1* obj = pypdsdata::Timepix::DataV1::pdsObject(self);
  if(not obj) return 0;
  
  // dimensions
  npy_intp dims[2] = { obj->height(), obj->width() };

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
  Pds::Timepix::DataV1* obj = pypdsdata::Timepix::DataV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "Timepix.DataV1(timestamp=" << obj->timestamp()
      << ", frameCounter=" << obj->frameCounter()
      << ", lostRows=" << obj->lostRows()
      << ", ...)" ;

  return PyString_FromString( str.str().c_str() );
}

}
