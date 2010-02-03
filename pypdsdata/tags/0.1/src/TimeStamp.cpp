//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeStamp...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TimeStamp.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int TimeStamp_init( PyObject* self, PyObject* args, PyObject* kwds );
  void TimeStamp_dealloc( PyObject* self );
  long TimeStamp_hash( PyObject* self );
  int TimeStamp_compare( PyObject *self, PyObject *other);
  PyObject* TimeStamp_str( PyObject *self );
  PyObject* TimeStamp_repr( PyObject *self );

  // type-specific methods
  PyObject* TimeStamp_ticks( PyObject* self );
  PyObject* TimeStamp_fiducials( PyObject* self );
  PyObject* TimeStamp_control( PyObject* self );
  PyObject* TimeStamp_vector( PyObject* self );

  PyMethodDef TimeStamp_Methods[] = {
    { "ticks", (PyCFunction) TimeStamp_ticks, METH_NOARGS, "Returns the ticks value" },
    { "fiducials", (PyCFunction) TimeStamp_fiducials, METH_NOARGS, "Returns the fiducials value" },
    { "control", (PyCFunction) TimeStamp_control, METH_NOARGS, "Returns the control value" },
    { "vector", (PyCFunction) TimeStamp_ticks, METH_NOARGS, "Returns the vector value" },
    {0, 0, 0, 0}
   };

  char TimeStamp_doc[] = "Python class wrapping C++ Pds::TimeStamp class.";

  PyTypeObject TimeStamp_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.TimeStamp",     /*tp_name*/
    sizeof(pypdsdata::TimeStamp), /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    TimeStamp_dealloc,       /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    TimeStamp_compare,       /*tp_compare*/
    TimeStamp_repr,          /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_TimeStamp*/
    0,                       /*tp_as_mapping*/
    TimeStamp_hash,          /*tp_hash*/
    0,                       /*tp_call*/
    TimeStamp_str,           /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    TimeStamp_doc,           /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    TimeStamp_Methods,       /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    TimeStamp_init,          /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    TimeStamp_dealloc        /*tp_del*/
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------


namespace pypdsdata {


PyTypeObject*
TimeStamp::typeObject()
{
  return &::TimeStamp_Type;
}

// makes new TimeStamp object from Pds type
PyObject*
TimeStamp::TimeStamp_FromPds(const Pds::TimeStamp& ts)
{
  pypdsdata::TimeStamp* ob = PyObject_New(pypdsdata::TimeStamp,&::TimeStamp_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create TimeStamp object." );
    return 0;
  }

  new(&ob->m_ts) Pds::TimeStamp(ts);

  return (PyObject*)ob;
}

} // namespace pypdsdata

namespace {

int
TimeStamp_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  int nArgs = PyTuple_GET_SIZE( args );
  if ( nArgs == 0 ) {

    new(&py_this->m_ts) Pds::TimeStamp();

  } else {

    unsigned ticks;
    unsigned fiducials;
    unsigned vector;
    unsigned control=0;
    if ( not PyArg_ParseTuple( args, "III|I:TimeStamp", &ticks, &fiducials, &vector, &control ) ) {
      return -1;
    }

    new(&py_this->m_ts) Pds::TimeStamp(ticks, fiducials, vector, control);
  }

  return 0;
}


void
TimeStamp_dealloc( PyObject* self )
{
  // deallocate ourself
  self->ob_type->tp_free(self);
}

long
TimeStamp_hash( PyObject* self )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  long hash = py_this->m_ts.fiducials();
  return hash;
}

int
TimeStamp_compare( PyObject* self, PyObject* other )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  pypdsdata::TimeStamp* py_other = (pypdsdata::TimeStamp*) other;
  if ( py_this->m_ts < py_other->m_ts ) return -1 ;
  if ( py_other->m_ts < py_this->m_ts ) return 1 ;
  return 0 ;
}

PyObject*
TimeStamp_str( PyObject *self )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  char buf[32];
  snprintf( buf, sizeof buf, "<%d:%d>", py_this->m_ts.ticks(),
      py_this->m_ts.fiducials() );
  return PyString_FromString( buf );
}

PyObject*
TimeStamp_repr( PyObject *self )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  char buf[64];
  snprintf( buf, sizeof buf, "<TimeStamp(%d,%d,%d,%d)>", py_this->m_ts.ticks(),
      py_this->m_ts.fiducials(), py_this->m_ts.vector(), py_this->m_ts.control() );
  return PyString_FromString( buf );
}

PyObject*
TimeStamp_ticks( PyObject* self )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  return PyInt_FromLong( py_this->m_ts.ticks() );
}

PyObject*
TimeStamp_fiducials( PyObject* self )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  return PyInt_FromLong( py_this->m_ts.fiducials() );
}

PyObject*
TimeStamp_control( PyObject* self )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  return PyInt_FromLong( py_this->m_ts.control() );
}

PyObject*
TimeStamp_vector( PyObject* self )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  return PyInt_FromLong( py_this->m_ts.vector() );
}

}
