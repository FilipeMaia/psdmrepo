//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ClockTime...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ClockTime.h"

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
  int ClockTime_init( PyObject* self, PyObject* args, PyObject* kwds );
  void ClockTime_dealloc( PyObject* self );
  long ClockTime_hash( PyObject* self );
  int ClockTime_compare( PyObject *self, PyObject *other);
  PyObject* ClockTime_str( PyObject *self );
  PyObject* ClockTime_repr( PyObject *self );

  // type-specific methods
  PyObject* ClockTime_seconds( PyObject* self, PyObject* );
  PyObject* ClockTime_nanoseconds( PyObject* self, PyObject* );

  PyMethodDef ClockTime_Methods[] = {
    { "seconds",     ClockTime_seconds,     METH_NOARGS, "Returns the number of seconds" },
    { "nanoseconds", ClockTime_nanoseconds, METH_NOARGS, "Returns the number of nanoseconds" },
    {0, 0, 0, 0}
   };

  char ClockTime_doc[] = "Python class wrapping C++ Pds::ClockTime class.\n\n"
      "Constructor takes two optional positional arguments - number of seconds\n"
      "and nanoseconds - both are integer numbers. If omitted they are\n"
      "initialized with zeros. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

  PyTypeObject ClockTime_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.ClockTime",     /*tp_name*/
    sizeof(pypdsdata::ClockTime), /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    ClockTime_dealloc,       /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    ClockTime_compare,       /*tp_compare*/
    ClockTime_repr,          /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_ClockTime*/
    0,                       /*tp_as_mapping*/
    ClockTime_hash,          /*tp_hash*/
    0,                       /*tp_call*/
    ClockTime_str,           /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    ClockTime_doc,           /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    ClockTime_Methods,       /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    ClockTime_init,          /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    ClockTime_dealloc        /*tp_del*/
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pypdsdata {


PyTypeObject*
ClockTime::typeObject()
{
  return &::ClockTime_Type;
}

// makes new ClockTime object from Pds type
PyObject*
ClockTime::ClockTime_FromPds(const Pds::ClockTime& clock)
{
  pypdsdata::ClockTime* ob = PyObject_New(pypdsdata::ClockTime,&::ClockTime_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create ClockTime object." );
    return 0;
  }

  new(&ob->m_clock) Pds::ClockTime(clock);

  return (PyObject*)ob;
}

} // namespace pypdsdata

namespace {

int
ClockTime_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned sec=0, nsec=0;
  if ( not PyArg_ParseTuple( args, "|II:ClockTime", &sec, &nsec ) ) return -1;

  if ( nsec >= 1000000000 ) {
    PyErr_SetString(PyExc_TypeError, "Error: nanoseconds value out of range");
    return -1;
  }

  new(&py_this->m_clock) Pds::ClockTime(sec, nsec);

  return 0;
}


void
ClockTime_dealloc( PyObject* self )
{
  // deallocate ourself
  self->ob_type->tp_free(self);
}

long
ClockTime_hash( PyObject* self )
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  long hash = py_this->m_clock.nanoseconds() + py_this->m_clock.seconds()*1000000000L;
  return hash;
}

int
ClockTime_compare( PyObject* self, PyObject* other )
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  pypdsdata::ClockTime* py_other = (pypdsdata::ClockTime*) other;
  if ( py_this->m_clock > py_other->m_clock ) return 1 ;
  if ( py_this->m_clock == py_other->m_clock ) return 0 ;
  return -1 ;
}

PyObject*
ClockTime_str( PyObject *self )
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  char buf[32];
  snprintf( buf, sizeof buf, "<T:%d.%09d>", py_this->m_clock.seconds(),
      py_this->m_clock.nanoseconds() );
  return PyString_FromString( buf );
}

PyObject*
ClockTime_repr( PyObject *self )
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  char buf[48];
  snprintf( buf, sizeof buf, "<ClockTime(%d, %d)>", py_this->m_clock.seconds(),
      py_this->m_clock.nanoseconds() );
  return PyString_FromString( buf );
}

PyObject*
ClockTime_seconds( PyObject* self, PyObject* )
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  return PyInt_FromLong( py_this->m_clock.seconds() );
}

PyObject*
ClockTime_nanoseconds( PyObject* self, PyObject* )
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  return PyInt_FromLong( py_this->m_clock.nanoseconds() );
}

}
