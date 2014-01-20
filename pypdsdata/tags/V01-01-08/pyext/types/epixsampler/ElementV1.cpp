//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ElementV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ElementV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

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
  FUN0_WRAPPER(pypdsdata::EpixSampler::ElementV1, vc)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ElementV1, lane)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ElementV1, acqCount)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ElementV1, frameNumber)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ElementV1, ticks)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ElementV1, fiducials)
  PyObject* frame( PyObject* self, PyObject* args );
  PyObject* temperatures( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "vc",            vc,            METH_NOARGS, "self.vc() -> int\n\nReturns integer number" },
    { "lane",          lane,          METH_NOARGS, "self.lane() -> int\n\nReturns integer number" },
    { "acqCount",      acqCount,      METH_NOARGS, "self.acqCount() -> int\n\nReturns integer number" },
    { "frameNumber",   frameNumber,   METH_NOARGS, "self.frameNumber() -> int\n\nReturns integer number" },
    { "ticks",         ticks,         METH_NOARGS, "self.ticks() -> int\n\nReturns integer number" },
    { "fiducials",     fiducials,     METH_NOARGS, "self.fiducials() -> int\n\nReturns integer number" },
    { "frame",         frame,         METH_VARARGS, "self.frame(config: ConfigV1) -> int\n\nReturns 2-dim array of integers" },
    { "temperatures",  temperatures,  METH_VARARGS, "self.temperatures(config: ConfigV1) -> int\n\nReturns 1-dim array of integers" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EpixSampler::ElementV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::EpixSampler::ElementV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ElementV1", module );
}

void
pypdsdata::EpixSampler::ElementV1::print(std::ostream& str) const
{
  str << "EpixSampler.ElementV1(vc=" << int(m_obj->vc())
      << ", lane=" << int(m_obj->lane())
      << ", acqCount=" << m_obj->acqCount()
      << ", frameNumber=" << m_obj->frameNumber()
      << ", ticks=" << m_obj->ticks()
      << ", fiducials=" << m_obj->fiducials()
      << ", ...)" ;
}

namespace {

PyObject*
frame( PyObject* self, PyObject* args )
{
  const Pds::EpixSampler::ElementV1* obj = pypdsdata::EpixSampler::ElementV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:EpixSampler.ElementV1.frame", &configObj ) ) return 0;

  // get config object from argument
  ndarray<const uint16_t, 2> data;
  if ( pypdsdata::EpixSampler::ConfigV1::Object_TypeCheck( configObj ) ) {
    Pds::EpixSampler::ConfigV1* config = pypdsdata::EpixSampler::ConfigV1::pdsObject( configObj );
    data = obj->frame(*config);
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a EpixSampler.ConfigV* object");
    return 0;
  }

  // NumPy type number
  int typenum = NPY_USHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  const unsigned* shape = data.shape();
  npy_intp dims[2] = { shape[0], shape[1] };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 2, dims, typenum, 0,
                                (void*)data.data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
temperatures( PyObject* self, PyObject* args )
{
  const Pds::EpixSampler::ElementV1* obj = pypdsdata::EpixSampler::ElementV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:EpixSampler.ElementV1.temperatures", &configObj ) ) return 0;

  // get config object from argument
  ndarray<const uint16_t, 1> data;;
  if ( pypdsdata::EpixSampler::ConfigV1::Object_TypeCheck( configObj ) ) {
    Pds::EpixSampler::ConfigV1* config = pypdsdata::EpixSampler::ConfigV1::pdsObject( configObj );
    data = obj->temperatures(*config);
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a EpixSampler.ConfigV* object");
    return 0;
  }

  // NumPy type number
  int typenum = NPY_USHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  const unsigned* shape = data.shape();
  npy_intp dims[1] = { shape[0] };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 1, dims, typenum, 0,
                                (void*)data.data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

}
