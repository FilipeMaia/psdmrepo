//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_DataDescV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataDescV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Exception.h"
#include "types/TypeLib.h"
#include "HorizV1.h"
#include "TimestampV1.h"
#include "pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "NumberOfBits",  Pds::Acqiris::DataDescV1::NumberOfBits },
        { "BitShift",      Pds::Acqiris::DataDescV1::BitShift },

        { 0, 0 }
  };

  // methods
  FUN0_WRAPPER(pypdsdata::Acqiris::DataDescV1, nbrSamplesInSeg)
  FUN0_WRAPPER(pypdsdata::Acqiris::DataDescV1, nbrSegments)
  FUN0_WRAPPER(pypdsdata::Acqiris::DataDescV1, indexFirstPoint)
  PyObject* timestamp( PyObject* self, PyObject* args );
  PyObject* waveform( PyObject* self, PyObject* args );
  PyObject* nextChannel( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    {"nbrSamplesInSeg",  nbrSamplesInSeg,   METH_NOARGS,  "Returns integer number" },
    {"nbrSegments",      nbrSegments,       METH_NOARGS,  "Returns integer number" },
    {"indexFirstPoint",  indexFirstPoint,   METH_NOARGS,  "Returns integer number" },
    {"timestamp",        timestamp,         METH_VARARGS, "Returns TimestampV1 object for a given segment" },
    {"waveform",         waveform,          METH_VARARGS, "Returns waveform array given a HorizV1 object" },
    {"nextChannel",      nextChannel,       METH_VARARGS, "Returns DataDescV1 for next channel (arg is HorizV1 object)" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::DataDescV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::DataDescV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "DataDescV1", module );
}

namespace {

PyObject*
timestamp( PyObject* self, PyObject* args )
{
  Pds::Acqiris::DataDescV1* obj = pypdsdata::Acqiris::DataDescV1::pdsObject( self );
  if ( not obj ) return 0;

  unsigned seg;
  if ( not PyArg_ParseTuple( args, "I:Acqiris.DataDescV1.timestamp", &seg ) ) return 0;

  return pypdsdata::Acqiris::TimestampV1::PyObject_FromPds( &obj->timestamp(seg), self );
}


PyObject*
waveform( PyObject* self, PyObject* args )
{
  Pds::Acqiris::DataDescV1* obj = pypdsdata::Acqiris::DataDescV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* hconfigObj ;
  if ( not PyArg_ParseTuple( args, "O:Acqiris.DataDescV1.waveform", &hconfigObj ) ) return 0;

  // check type
  if ( not pypdsdata::Acqiris::HorizV1::Object_TypeCheck( hconfigObj ) ) {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a Acqiris.HorizV1 object");
    return 0;
  }

  // convert to Pds config object
  const Pds::Acqiris::HorizV1* hconfig = pypdsdata::Acqiris::HorizV1::pdsObject( hconfigObj );

  // get data size
  unsigned size = hconfig->nbrSamples();

  // NumPy type number
  int typenum = NPY_SHORT;

  // not writable
  int flags = NPY_C_CONTIGUOUS ;

  // dimensions
  npy_intp dims[1] = { size };

  // make array
  int16_t* data = obj->waveform(*hconfig);
  data += obj->indexFirstPoint();
  PyObject* array = PyArray_New(&PyArray_Type, 1, dims, typenum, 0,
                                (void*)data, 0, flags, 0);

  // array does not own its data, set self as owner
  Py_INCREF(self);
  PyArrayObject* oarray = (PyArrayObject*)array;
  oarray->base = self ;

  return array;
}

PyObject*
nextChannel( PyObject* self, PyObject* args )
{
  Pds::Acqiris::DataDescV1* obj = pypdsdata::Acqiris::DataDescV1::pdsObject( self );
  if ( not obj ) return 0;

  // parse args
  PyObject* hconfigObj ;
  if ( not PyArg_ParseTuple( args, "O:Acqiris.DataDescV1.nextChannel", &hconfigObj ) ) return 0;

  // check type
  if ( not pypdsdata::Acqiris::HorizV1::Object_TypeCheck( hconfigObj ) ) {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a Acqiris.HorizV1 object");
    return 0;
  }

  // convert to Pds config object
  const Pds::Acqiris::HorizV1* hconfig = pypdsdata::Acqiris::HorizV1::pdsObject( hconfigObj );

  // get next object
  Pds::Acqiris::DataDescV1* next = obj->nextChannel( *hconfig );

  // make Python object
  pypdsdata::Acqiris::DataDescV1* py_this = (pypdsdata::Acqiris::DataDescV1*) self;
  return pypdsdata::Acqiris::DataDescV1::PyObject_FromPds( next, py_this->m_parent, py_this->m_dtor );
}


}

