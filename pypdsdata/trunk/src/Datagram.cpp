//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Datagram...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Datagram.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Exception.h"
#include "Sequence.h"
#include "Xtc.h"
#include "pdsdata/xtc/Env.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int Datagram_init( PyObject* self, PyObject* args, PyObject* kwds );
  void Datagram_dealloc( PyObject* self );

  // type-specific methods
  PyObject* Datagram_env( PyObject* self );
  PyObject* Datagram_seq( PyObject* self );
  PyObject* Datagram_xtc( PyObject* self );

  PyMethodDef Datagram_Methods[] = {
    { "env", (PyCFunction) Datagram_env, METH_NOARGS, "Returns the env field as a number." },
    { "seq", (PyCFunction) Datagram_seq, METH_NOARGS, "Returns the seq field as an object." },
    { "xtc", (PyCFunction) Datagram_xtc, METH_NOARGS, "Returns top-level Xtc object." },
    {0, 0, 0, 0}
   };

  char Datagram_doc[] = "Python class wrapping C++ Pds::Dgram class.\n\n"
      "Instances of his class are created by other objects, there is no\n"
      "sensible constructor for now that can be used at Python level.";

  PyTypeObject Datagram_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.Datagram",      /*tp_name*/
    sizeof(pypdsdata::Datagram), /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    Datagram_dealloc,        /*tp_dealloc*/
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
    Datagram_doc,            /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    Datagram_Methods,        /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    Datagram_init,           /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    Datagram_dealloc         /*tp_del*/
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pypdsdata {

PyTypeObject*
Datagram::typeObject()
{
  return &::Datagram_Type;
}

/// Builds Datagram from corresponding Pds type, parent is the owner
/// of the corresponding buffer space, if parent is 0 then datagram
/// will be deleted on destruction.
PyObject*
Datagram::Datagram_FromPds( Pds::Dgram* object, PyObject* parent, destructor dtor )
{
  pypdsdata::Datagram* ob = PyObject_New(pypdsdata::Datagram,&::Datagram_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create Datagram object." );
    return 0;
  }

  ob->m_object = object;
  ob->m_parent = parent ;
  Py_XINCREF(ob->m_parent);
  ob->m_dtor = dtor;

  return (PyObject*)ob;
}

} // namespace pypdsdata


namespace {

int
Datagram_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::Datagram* py_this = (pypdsdata::Datagram*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  py_this->m_object = 0;
  py_this->m_parent = 0;
  py_this->m_dtor = 0;

  return 0;
}


void
Datagram_dealloc( PyObject* self )
{
  pypdsdata::Datagram* py_this = (pypdsdata::Datagram*) self;

  // if destructor is defined then call it
  if ( py_this->m_dtor ) {
    (*py_this->m_dtor)( py_this->m_object ) ;
    py_this->m_object = 0;
  }

  // free parent from us
  Py_CLEAR(py_this->m_parent);

  // deallocate ourself
  self->ob_type->tp_free(self);
}

PyObject*
Datagram_env( PyObject* self )
{
  pypdsdata::Datagram* py_this = (pypdsdata::Datagram*) self;
  if( ! py_this->m_object ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return PyInt_FromLong( py_this->m_object->env.value() );
}

PyObject*
Datagram_seq( PyObject* self )
{
  pypdsdata::Datagram* py_this = (pypdsdata::Datagram*) self;
  if( ! py_this->m_object ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::Sequence::Sequence_FromPds( py_this->m_object->seq );
}

PyObject*
Datagram_xtc( PyObject* self )
{
  pypdsdata::Datagram* py_this = (pypdsdata::Datagram*) self;
  if( ! py_this->m_object ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::Xtc::Xtc_FromPds( &py_this->m_object->xtc, self, 0 );
}

}
