//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Sequence...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Sequence.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ClockTime.h"
#include "TimeStamp.h"
#include "TransitionId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int Sequence_init( PyObject* self, PyObject* args, PyObject* kwds );
  void Sequence_dealloc( PyObject* self );

  // type-specific methods
  PyObject* Sequence_type( PyObject* self );
  PyObject* Sequence_service( PyObject* self );
  PyObject* Sequence_isExtended( PyObject* self );
  PyObject* Sequence_isEvent( PyObject* self );
  PyObject* Sequence_clock( PyObject* self );
  PyObject* Sequence_stamp( PyObject* self );

  PyMethodDef Sequence_Methods[] = {
    { "type", (PyCFunction) Sequence_type, METH_NOARGS,
        "Returns the type of this sequence, one of Event, Occurrence, or Marker" },
    { "service", (PyCFunction) Sequence_service, METH_NOARGS, "Returns the TransitionId type" },
    { "isExtended", (PyCFunction) Sequence_isExtended, METH_NOARGS, "Returns True for extended sequence" },
    { "isEvent", (PyCFunction) Sequence_isEvent, METH_NOARGS, "Returns True for event sequence" },
    { "clock", (PyCFunction) Sequence_clock, METH_NOARGS, "Returns clock value for sequence" },
    { "stamp", (PyCFunction) Sequence_stamp, METH_NOARGS, "Returns timestamp value for sequence" },
    {0, 0, 0, 0}
   };

  char Sequence_doc[] = "Python class wrapping C++ Pds::Sequence class.";

  PyTypeObject Sequence_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.Sequence",      /*tp_name*/
    sizeof(pypdsdata::Sequence), /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    Sequence_dealloc,        /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    0,                       /*tp_compare*/
    0,                       /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_sequence*/
    0,                       /*tp_as_mapping*/
    0,                       /*tp_hash*/
    0,                       /*tp_call*/
    0,                       /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    Sequence_doc,            /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    Sequence_Methods,        /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    Sequence_init,           /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    Sequence_dealloc         /*tp_del*/
  };

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pypdsdata {


PyTypeObject*
Sequence::typeObject()
{
  static bool once = true;
  if (once) {
    once = false;

    // define class attributes for enums
    PyObject* tp_dict = PyDict_New();
    PyDict_SetItemString( tp_dict, "Event", PyInt_FromLong(Pds::Sequence::Event) );
    PyDict_SetItemString( tp_dict, "Occurrence", PyInt_FromLong(Pds::Sequence::Occurrence) );
    PyDict_SetItemString( tp_dict, "Marker", PyInt_FromLong(Pds::Sequence::Marker) );
    ::Sequence_Type.tp_dict = tp_dict;
  }

  return &::Sequence_Type;
}

// makes new Sequence object from Pds type
PyObject*
Sequence::Sequence_FromPds(const Pds::Sequence& seq)
{
  pypdsdata::Sequence* ob = PyObject_New(pypdsdata::Sequence,&::Sequence_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create Sequence object." );
    return 0;
  }

  new(&ob->m_seq) Pds::Sequence(seq);

  return (PyObject*)ob;
}

} // namespace pypdsdata

namespace {

int
Sequence_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::Sequence* py_this = (pypdsdata::Sequence*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  new(&py_this->m_seq) Pds::Sequence();

  return 0;
}


void
Sequence_dealloc( PyObject* self )
{
  // deallocate ourself
  self->ob_type->tp_free(self);
}

PyObject*
Sequence_type( PyObject* self )
{
  pypdsdata::Sequence* py_this = (pypdsdata::Sequence*) self;
  return PyInt_FromLong( py_this->m_seq.type() );
}

PyObject*
Sequence_service( PyObject* self )
{
  pypdsdata::Sequence* py_this = (pypdsdata::Sequence*) self;
  return pypdsdata::TransitionId::TransitionId_FromInt( py_this->m_seq.service() );
}

PyObject*
Sequence_isExtended( PyObject* self )
{
  pypdsdata::Sequence* py_this = (pypdsdata::Sequence*) self;
  return PyBool_FromLong( py_this->m_seq.isExtended() );
}

PyObject*
Sequence_isEvent( PyObject* self )
{
  pypdsdata::Sequence* py_this = (pypdsdata::Sequence*) self;
  return PyBool_FromLong( py_this->m_seq.isEvent() );
}

PyObject*
Sequence_clock( PyObject* self )
{
  pypdsdata::Sequence* py_this = (pypdsdata::Sequence*) self;
  return pypdsdata::ClockTime::ClockTime_FromPds( py_this->m_seq.clock() );
}

PyObject*
Sequence_stamp( PyObject* self )
{
  pypdsdata::Sequence* py_this = (pypdsdata::Sequence*) self;
  return pypdsdata::TimeStamp::TimeStamp_FromPds( py_this->m_seq.stamp() );
}

}
