//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcIterator...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcIterator.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Xtc.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int XtcIterator_init( PyObject* self, PyObject* args, PyObject* kwds );
  void XtcIterator_dealloc( PyObject* self );

  // type-specific methods
  PyObject* XtcIterator_iter( PyObject* self );
  PyObject* XtcIterator_next( PyObject* self );

  PyMethodDef XtcIterator_Methods[] = {
    {0, 0, 0, 0}
   };

  char XtcIterator_doc[] = "Python class wrapping C++ Pds::XtcIterator class.\n\n"
      "Constructor of the class has one argument which is a Python XTC object. The\n"
      "instances of this class act as regular Python iterators which return objects\n"
      "of type Xtc.";

  PyTypeObject XtcIterator_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.XtcIterator",   /*tp_name*/
    sizeof(pypdsdata::XtcIterator), /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    XtcIterator_dealloc,     /*tp_dealloc*/
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
    XtcIterator_doc,         /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    XtcIterator_iter,        /*tp_iter*/
    XtcIterator_next,        /*tp_iternext*/
    XtcIterator_Methods,     /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    XtcIterator_init,        /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    XtcIterator_dealloc      /*tp_del*/
  };


}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace pypdsdata {

PyTypeObject*
XtcIterator::typeObject()
{
  return &::XtcIterator_Type;
}

PyObject*
XtcIterator::XtcIterator_FromXtc( Pds::Xtc* xtc, PyObject* parent )
{
  // check XTC first
  if ( xtc->contains.id() != Pds::TypeId::Id_Xtc ) {
    PyErr_SetString(PyExc_TypeError, "Error: XtcIterator cannot make iterator for non-Xtc container");
    return 0;
  }

  pypdsdata::XtcIterator* py_this = PyObject_New(pypdsdata::XtcIterator,&::XtcIterator_Type);
  if ( not py_this ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create XtcIterator object." );
    return 0;
  }

  // build an object
  py_this->m_parentXtc = xtc;
  Py_INCREF(parent);
  py_this->m_parent = parent;
  py_this->m_remaining = xtc->sizeofPayload();
  if ( py_this->m_remaining ) {
    py_this->m_next = (Pds::Xtc*)(xtc->payload());
  } else {
    py_this->m_next = 0;
  }

  return (PyObject*)py_this;
}

} // namespace pypdsdata


namespace {

int
XtcIterator_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::XtcIterator* py_this = (pypdsdata::XtcIterator*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  PyObject* xtcObj ;
  if ( not PyArg_ParseTuple( args, "O:XtcIterator", &xtcObj ) ) return -1;

  if ( not pypdsdata::Xtc::Xtc_Check( xtcObj ) ) {
    PyErr_SetString(PyExc_TypeError, "Error: XtcIterator expects Xtc object");
    return -1;
  }

  // convert and check again
  Pds::Xtc* xtc = pypdsdata::Xtc::Xtc_AsPds( xtcObj );
  if ( not xtc ) {
    PyErr_SetString(PyExc_TypeError, "Error: XtcIterator expects Xtc object");
    return -1;
  }

  // check xtc contains type
  if ( xtc->contains.id() != Pds::TypeId::Id_Xtc ) {
    PyErr_SetString(PyExc_TypeError, "Error: XtcIterator cannot make iterator for non-Xtc container");
    return -1;
  }

  // build an object
  py_this->m_parentXtc = xtc;
  Py_INCREF(xtcObj);
  py_this->m_parent = xtcObj;
  py_this->m_remaining = py_this->m_parentXtc->sizeofPayload();
  if ( py_this->m_remaining ) {
    py_this->m_next = (Pds::Xtc*)(py_this->m_parentXtc->payload());
  } else {
    py_this->m_next = 0;
  }

  return 0;
}


void
XtcIterator_dealloc( PyObject* self )
{
  pypdsdata::XtcIterator* py_this = (pypdsdata::XtcIterator*) self;

  // free the parent from us
  Py_XDECREF(py_this->m_parent);

  // deallocate ourself
  self->ob_type->tp_free(self);
}


PyObject*
XtcIterator_iter( PyObject* self )
{
  Py_XINCREF(self);
  return self;
}

PyObject*
XtcIterator_next( PyObject* self )
{
  pypdsdata::XtcIterator* py_this = (pypdsdata::XtcIterator*) self;

  Pds::Xtc* next = py_this->m_next;

  if ( not next ) {
    // stop it
    PyErr_SetNone( PyExc_StopIteration );
    return 0;
  }

  // advance
  py_this->m_remaining -= next->extent;
  if ( py_this->m_remaining > 0 ) {
    py_this->m_next = next->next();
  } else {
    py_this->m_next = 0;
  }

  return pypdsdata::Xtc::PyObject_FromPds( next, py_this->m_parent, 0 );
}

}
