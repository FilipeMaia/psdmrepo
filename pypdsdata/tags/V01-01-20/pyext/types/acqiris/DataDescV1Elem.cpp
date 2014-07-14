//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_DataDescV1Elem...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataDescV1Elem.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"
#include "ConfigV1.h"
#include "TimestampV1.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "NumberOfBits",  Pds::Acqiris::DataDescV1Elem::NumberOfBits },
        { "BitShift",      Pds::Acqiris::DataDescV1Elem::BitShift },

        { 0, 0 }
  };

  // methods
  FUN0_WRAPPER(pypdsdata::Acqiris::DataDescV1Elem, nbrSamplesInSeg)
  FUN0_WRAPPER(pypdsdata::Acqiris::DataDescV1Elem, nbrSegments)
  FUN0_WRAPPER(pypdsdata::Acqiris::DataDescV1Elem, indexFirstPoint)
  PyObject* timestamp( PyObject* self, PyObject* args );
  PyObject* waveforms( PyObject* self, PyObject* args );
  PyObject* nextChannel( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"nbrSamplesInSeg",  nbrSamplesInSeg,   METH_NOARGS,  "self.nbrSamplesInSeg() -> int\n\nReturns integer number" },
    {"nbrSegments",      nbrSegments,       METH_NOARGS,  "self.nbrSegments() -> int\n\nReturns integer number" },
    {"indexFirstPoint",  indexFirstPoint,   METH_NOARGS,  "self.indexFirstPoint() -> int\n\nReturns integer number" },
    {"timestamp",        timestamp,         METH_VARARGS,
        "self.timestamp(cfg: ConfigV1) -> list\n\nReturns the list of :py:class:`TimestampV1` objects, "
        "requires a :py:class:`ConfigV1` object as an argument." },
    {"waveforms",         waveforms,          METH_VARARGS,
        "self.waveforms(cfg: ConfigV1) -> numpy.ndarray\n\nReturns waveform array given a :py:class:`ConfigV1` object." },
    {"waveform",         waveforms,          METH_VARARGS,
        "self.waveform(cfg: ConfigV1) -> numpy.ndarray\n\nAlias for waveforms() method." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::DataDescV1Elem class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::DataDescV1Elem::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "DataDescV1Elem", module );
}

void 
pypdsdata::Acqiris::DataDescV1Elem::print(std::ostream& out) const
{
  if(not m_obj) {
    out << "acqiris.DataDescV1Elem(None)";
  } else {  
    out << "acqiris.DataDescV1Elem(nbrSegments=" << m_obj->nbrSegments()
        << ", nbrSamplesInSeg=" << m_obj->nbrSamplesInSeg() 
        << ", ...)";
  }
}

namespace {

PyObject*
timestamp( PyObject* self, PyObject* args )
{
  Pds::Acqiris::DataDescV1Elem* obj = pypdsdata::Acqiris::DataDescV1Elem::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* pyconfig ;
  if ( not PyArg_ParseTuple( args, "O:Acqiris.DataDescV1Elem.waveform", &pyconfig ) ) return 0;

  // check type
  if ( not pypdsdata::Acqiris::ConfigV1::Object_TypeCheck( pyconfig ) ) {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a Acqiris.ConfigV1 object");
    return 0;
  }

  // convert to Pds config object
  const Pds::Acqiris::ConfigV1* config = pypdsdata::Acqiris::ConfigV1::pdsObject( pyconfig );

  const ndarray<const Pds::Acqiris::TimestampV1, 1>& timestamp = obj->timestamp(*config);

  // if argument is missing the return list of objects, otherwise return single object
  using pypdsdata::TypeLib::toPython;
  return toPython(timestamp);
}


PyObject*
waveforms( PyObject* self, PyObject* args )
{
  Pds::Acqiris::DataDescV1Elem* obj = pypdsdata::Acqiris::DataDescV1Elem::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* pyconfig ;
  if ( not PyArg_ParseTuple( args, "O:Acqiris.DataDescV1Elem.waveform", &pyconfig ) ) return 0;

  // check type
  if ( not pypdsdata::Acqiris::ConfigV1::Object_TypeCheck( pyconfig ) ) {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a Acqiris.ConfigV1 object");
    return 0;
  }

  // convert to Pds config object
  const Pds::Acqiris::ConfigV1* config = pypdsdata::Acqiris::ConfigV1::pdsObject( pyconfig );

  // NumPy type number
  int typenum = NPY_SHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  ndarray<const int16_t, 2> waveforms = obj->waveforms(*config);

  // dimensions
  npy_intp dims[2] = { waveforms.shape()[0], waveforms.shape()[1] };

  // make array
  const int16_t* data = waveforms.data();
  //data += obj->indexFirstPoint(); -- not needed, already corrected
  PyObject* array = PyArray_New(&PyArray_Type, 2, dims, typenum, 0,
                                (void*)data, 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  PyArrayObject* oarray = (PyArrayObject*)array;
  oarray->base = self ;

  return array;
}

}

