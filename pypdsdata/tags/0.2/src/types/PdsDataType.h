#ifndef PYPDSDATA_PDSDATATYPE_H
#define PYPDSDATA_PDSDATATYPE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PdsDataType.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "Python.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {

/**
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

template <typename ConcreteType, typename PdsType>
struct PdsDataType {

  // type of the destructor function
  typedef void (*destructor)(const PdsType*);

  /// Builds Python object from corresponding Pds type, parent is the owner
  /// of the corresponding buffer space, usually XTC object. If destructor
  /// function is provided it will be called to delete the Pds object.
  static PyObject* PyObject_FromPds( const PdsType* obj, PyObject* parent, destructor dtor=0 );

  static const PdsType* pdsObject(PyObject* self) {
    PdsDataType* py_this = (PdsDataType*) self;
    if( ! py_this->m_obj ){
      PyErr_SetString(PyExc_TypeError, "Error: No Valid C++ Object");
    }
    return py_this->m_obj;
  }

  // --------------------------------------------------

  // standard Python stuff
  PyObject_HEAD

  const PdsType* m_obj;
  PyObject* m_parent;
  destructor m_dtor;

protected:

  static PyTypeObject* PdsDataType_typeObject();
  static void PdsDataType_dealloc( PyObject* self );

};


/// Returns the Python type opbject
template <typename ConcreteType, typename PdsType>
PyTypeObject*
PdsDataType<ConcreteType, PdsType>::PdsDataType_typeObject()
{
  static PyTypeObject type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    0,                       /*tp_name*/
    sizeof(ConcreteType),    /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    PdsDataType_dealloc,     /*tp_dealloc*/
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
    0,                       /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    0,                       /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    0,                       /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    PdsDataType_dealloc      /*tp_del*/
  };

  return &type;
}

template <typename ConcreteType, typename PdsType>
void
PdsDataType<ConcreteType, PdsType>::PdsDataType_dealloc( PyObject* self )
{
  PdsDataType* py_this = (PdsDataType*) self;

  // if destructor is defined then call it
  if ( py_this->m_dtor ) {
    (*py_this->m_dtor)( py_this->m_obj ) ;
    py_this->m_obj = 0;
  }

  // free parent from us
  Py_CLEAR(py_this->m_parent);

  // deallocate ourself
  self->ob_type->tp_free(self);
}

/// Builds Python object from corresponding Pds type, parent is the owner
/// of the corresponding buffer space, usually XTC object. If destructor
/// function is provided it will be called to delete the Pds object.
template <typename ConcreteType, typename PdsType>
PyObject*
PdsDataType<ConcreteType, PdsType>::PyObject_FromPds( const PdsType* obj, PyObject* parent, destructor dtor )
{
  ConcreteType* ob = PyObject_New(ConcreteType,ConcreteType::typeObject());
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create PdsDataType object." );
    return 0;
  }

  ob->m_obj = obj;
  ob->m_parent = parent ;
  Py_XINCREF(ob->m_parent);
  ob->m_dtor = dtor;

  return (PyObject*)ob;
}


} // namespace pypdsdata

#endif // PYPDSDATA_PDSDATATYPE_H
