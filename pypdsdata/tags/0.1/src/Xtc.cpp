//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Xtc...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Xtc.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Damage.h"
#include "Exception.h"
#include "TypeId.h"
#include "XtcIterator.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int Xtc_init( PyObject* self, PyObject* args, PyObject* kwds );
  void Xtc_dealloc( PyObject* self );
  PyObject* Xtc_iter( PyObject* self );

  // type-specific methods
  PyObject* Xtc_damage( PyObject* self );
  PyObject* Xtc_contains( PyObject* self );
  PyObject* Xtc_extent( PyObject* self );
  PyObject* Xtc_sizeofPayload( PyObject* self );
  PyObject* Xtc_payload( PyObject* self );

  PyMethodDef Xtc_Methods[] = {
    { "damage", (PyCFunction) Xtc_damage, METH_NOARGS, "Returns the damage field." },
    { "contains", (PyCFunction) Xtc_contains, METH_NOARGS, "Returns the TypeId of the contained object(s)." },
    { "extent", (PyCFunction) Xtc_extent, METH_NOARGS, "Returns the extent size of the Xtc." },
    { "sizeofPayload", (PyCFunction) Xtc_sizeofPayload, METH_NOARGS, "Returns the size of payload." },
    { "payload", (PyCFunction) Xtc_payload, METH_NOARGS, "Returns data object. If `contains' is Any returns None. If `contains' is Id_Xtc returns XtcIterator" },
    {0, 0, 0, 0}
   };

  char Xtc_doc[] = "Python class wrapping C++ Pds::Xtc class.";

  PyTypeObject Xtc_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.Xtc",           /*tp_name*/
    sizeof(pypdsdata::Xtc),  /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    Xtc_dealloc,             /*tp_dealloc*/
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
    Xtc_doc,                 /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    Xtc_iter,                /*tp_iter*/
    0,                       /*tp_iternext*/
    Xtc_Methods,             /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    Xtc_init,                /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    Xtc_dealloc              /*tp_del*/
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------


namespace pypdsdata {

PyTypeObject*
Xtc::typeObject()
{
  return &::Xtc_Type;
}

/// Builds Xtc from corresponding Pds type, parent is the owner
/// of the corresponding buffer space, if parent is 0 then Xtc
/// will be deleted on destruction.
PyObject*
Xtc::Xtc_FromPds( Pds::Xtc* xtc, PyObject* parent, destructor dtor )
{
  pypdsdata::Xtc* ob = PyObject_New(pypdsdata::Xtc,&::Xtc_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create Xtc object." );
    return 0;
  }

  ob->m_xtc = xtc;
  ob->m_parent = parent ;
  Py_XINCREF(ob->m_parent);
  ob->m_dtor = dtor;

  return (PyObject*)ob;
}

// Check object type
bool
Xtc::Xtc_Check( PyObject* obj )
{
  return PyObject_TypeCheck( obj, &::Xtc_Type );
}

// REturns a pointer to Pds object
Pds::Xtc*
Xtc::Xtc_AsPds( PyObject* obj )
{
  if ( obj->ob_type == &::Xtc_Type ) {
    return ((pypdsdata::Xtc*)obj)->m_xtc;
  } else {
    return 0;
  }
}


} // namespace pypdsdata


namespace {

int
Xtc_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  py_this->m_xtc = 0;
  py_this->m_parent = 0;
  py_this->m_dtor = 0;

  return 0;
}


void
Xtc_dealloc( PyObject* self )
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;

  // if destructor is defined then call it
  if ( py_this->m_dtor ) {
    (*py_this->m_dtor)( py_this->m_xtc ) ;
    py_this->m_xtc = 0;
  }

  // free parent from us
  Py_CLEAR(py_this->m_parent);

  // deallocate ourself
  self->ob_type->tp_free(self);
}


PyObject*
Xtc_iter( PyObject* self )
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if( ! py_this->m_xtc ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  // it will throw exception if this xtc of incorrect type
  return pypdsdata::XtcIterator::XtcIterator_FromXtc( py_this->m_xtc, self );
}

PyObject*
Xtc_damage( PyObject* self )
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if( ! py_this->m_xtc ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::Damage::Damage_FromInt( py_this->m_xtc->damage.value() );
}

PyObject*
Xtc_contains( PyObject* self )
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if( ! py_this->m_xtc ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::TypeId::TypeId_FromPds( py_this->m_xtc->contains );
}

PyObject*
Xtc_extent( PyObject* self )
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if( ! py_this->m_xtc ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return PyInt_FromLong( py_this->m_xtc->extent );
}

PyObject*
Xtc_sizeofPayload( PyObject* self )
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if( ! py_this->m_xtc ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return PyInt_FromLong( py_this->m_xtc->sizeofPayload() );
}

PyObject*
Xtc_payload( PyObject* self )
{
  pypdsdata::Xtc* py_this = (pypdsdata::Xtc*) self;
  if( ! py_this->m_xtc ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  if ( py_this->m_xtc->contains.id() == Pds::TypeId::Id_Xtc ) {
    return Xtc_iter( self );
  } else {
    PyErr_SetString(PyExc_TypeError, "Error: payload() does not work yet on non-XTC container");
    return 0;
  }
}

}
