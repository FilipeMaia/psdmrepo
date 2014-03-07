//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataV2.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigV2.h"
#include "../TypeLib.h"
#include "../../EnumType.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "iDataReadSize", Pds::OceanOptics::DataV2::iDataReadSize },

        { "iNumPixels", Pds::OceanOptics::DataV2::iNumPixels },

        { "iActivePixelIndex", Pds::OceanOptics::DataV2::iActivePixelIndex },

        { 0, 0 }
  };

  // methods
  FUN0_WRAPPER(pypdsdata::OceanOptics::DataV2, frameCounter)
  FUN0_WRAPPER(pypdsdata::OceanOptics::DataV2, numDelayedFrames)
  FUN0_WRAPPER(pypdsdata::OceanOptics::DataV2, numDiscardFrames)
  FUN0_WRAPPER(pypdsdata::OceanOptics::DataV2, numSpectraInData)
  FUN0_WRAPPER(pypdsdata::OceanOptics::DataV2, numSpectraInQueue)
  FUN0_WRAPPER(pypdsdata::OceanOptics::DataV2, numSpectraUnused)
  FUN0_WRAPPER(pypdsdata::OceanOptics::DataV2, durationOfFrame)
  PyObject* data( PyObject* self, PyObject* );
  PyObject* timeFrameStart( PyObject* self, PyObject* );
  PyObject* timeFrameFirstData( PyObject* self, PyObject* );
  PyObject* timeFrameEnd( PyObject* self, PyObject* );
  PyObject* nonlinerCorrected( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"frameCounter",        frameCounter,       METH_NOARGS,  "self.frameCounter() -> int\n\nReturns integer number." },
    {"numDelayedFrames",    numDelayedFrames,   METH_NOARGS,  "self.numDelayedFrames() -> int\n\nReturns integer number." },
    {"numDiscardFrames",    numDiscardFrames,   METH_NOARGS,  "self.numDiscardFrames() -> int\n\nReturns integer number." },
    {"numSpectraInData",    numSpectraInData,   METH_NOARGS,  "self.numSpectraInData() -> int\n\nReturns integer number." },
    {"numSpectraInQueue",   numSpectraInQueue,  METH_NOARGS,  "self.numSpectraInQueue() -> int\n\nReturns integer number." },
    {"numSpectraUnused",    numSpectraUnused,   METH_NOARGS,  "self.numSpectraUnused() -> int\n\nReturns integer number." },
    {"durationOfFrame",     durationOfFrame,    METH_NOARGS,  "self.durationOfFrame() -> float\n\nReturns floating number." },
    {"data",                data,               METH_NOARGS,
        "self.data() -> ndarray\n\nReturns 1-dim ndarray of integers, size of array is `iNumPixels`" },
    {"timeFrameStart",      timeFrameStart,     METH_NOARGS,
        "self.timeFrameStart() -> tuple\n\nReturns tuple of two integers (sec, nsec)." },
    {"timeFrameFirstData",  timeFrameFirstData, METH_NOARGS,
        "self.timeFrameFirstData() -> tuple\n\nReturns tuple of two integers (sec, nsec)." },
    {"timeFrameEnd",        timeFrameEnd,       METH_NOARGS,
        "self.timeFrameEnd() -> tuple\n\nReturns tuple of two integers (sec, nsec)." },
    {"nonlinerCorrected",   nonlinerCorrected,  METH_VARARGS,
        "self.nonlinCorrect(cfg: ConfigVx) -> ndarray\n\nReturns 1-dim ndarray of floats, which is data corrected for non-linearity,"
        " size of array is `iNumPixels`" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::OceanOptics::DataV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::OceanOptics::DataV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "DataV2", module );
}

void
pypdsdata::OceanOptics::DataV2::print(std::ostream& str) const
{
  str << "oceanoptics.DataV2(frameCounter=" << m_obj->frameCounter()
      << ", numDelayedFrames=" << m_obj->numDelayedFrames()
      << ", numDiscardFrames=" << m_obj->numDiscardFrames()
      << ", durationOfFrame=" << m_obj->durationOfFrame()
      << ", ...)" ;
}

namespace {

PyObject*
data( PyObject* self, PyObject* )
{
  Pds::OceanOptics::DataV2* obj = pypdsdata::OceanOptics::DataV2::pdsObject(self);
  if ( not obj ) return 0;

  // NumPy type number
  int typenum = NPY_USHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  npy_intp dims[1] = { Pds::OceanOptics::DataV2::iNumPixels };

  // make array
  PyObject* array = PyArray_New(&PyArray_Type, 1, dims, typenum, 0,
                                (void*)obj->data().data(), 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  ((PyArrayObject*)array)->base = self ;

  return array;
}

PyObject*
timeFrameStart( PyObject* self, PyObject* )
{
  Pds::OceanOptics::DataV2* obj = pypdsdata::OceanOptics::DataV2::pdsObject(self);
  if(not obj) return 0;

  const Pds::OceanOptics::timespec64& ts = obj->timeFrameStart();
  PyObject* tuple = PyTuple_New(2);
  PyTuple_SET_ITEM(tuple, 0, PyLong_FromLongLong(ts.tv_sec()));
  PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(ts.tv_nsec()));

  return tuple;
}

PyObject*
timeFrameFirstData( PyObject* self, PyObject* )
{
  Pds::OceanOptics::DataV2* obj = pypdsdata::OceanOptics::DataV2::pdsObject(self);
  if(not obj) return 0;

  const Pds::OceanOptics::timespec64& ts = obj->timeFrameFirstData();
  PyObject* tuple = PyTuple_New(2);
  PyTuple_SET_ITEM(tuple, 0, PyLong_FromLongLong(ts.tv_sec()));
  PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(ts.tv_nsec()));

  return tuple;
}

PyObject*
timeFrameEnd( PyObject* self, PyObject* )
{
  Pds::OceanOptics::DataV2* obj = pypdsdata::OceanOptics::DataV2::pdsObject(self);
  if(not obj) return 0;

  const Pds::OceanOptics::timespec64& ts = obj->timeFrameEnd();
  PyObject* tuple = PyTuple_New(2);
  PyTuple_SET_ITEM(tuple, 0, PyLong_FromLongLong(ts.tv_sec()));
  PyTuple_SET_ITEM(tuple, 1, PyInt_FromLong(ts.tv_nsec()));

  return tuple;
}

PyObject*
nonlinerCorrected( PyObject* self, PyObject* args)
{
  Pds::OceanOptics::DataV2* obj = pypdsdata::OceanOptics::DataV2::pdsObject(self);
  if ( not obj ) return 0;

  // parse args
  PyObject* configObj ;
  if ( not PyArg_ParseTuple( args, "O:OceanOptics.DataV2.nonlinerCorrected", &configObj ) ) return 0;

  // get Pds::OceanOptics::ConfigV2 from argument
  const Pds::OceanOptics::ConfigV2* config = 0;
  if ( pypdsdata::OceanOptics::ConfigV2::Object_TypeCheck( configObj ) ) {
    config = pypdsdata::OceanOptics::ConfigV2::pdsObject( configObj );
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a oceanoptics.ConfigV2 object");
    return 0;
  }


  // NumPy type number
  int typenum = NPY_DOUBLE;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  npy_intp dims[1] = { Pds::OceanOptics::DataV2::iNumPixels };

  // make array, it will allocate memory for its data
  PyObject* array = PyArray_New(&PyArray_Type, 1, dims, typenum, 0,
                                (void*)0, 0, flags, 0);

  // copy corrected data
  for (int i = 0; i != Pds::OceanOptics::DataV2::iNumPixels; ++ i) {
    *(double*)PyArray_GETPTR1(array, i) = obj->nonlinerCorrected(*config, i);
  }

  return array;
}

}
