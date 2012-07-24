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
#include "python/Python.h"
#include <string>
#include <iostream>
#include <sstream>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Xtc.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "../Exception.h"

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
struct PdsDataType : PyObject {

  // type of the destructor function
  typedef void (*destructor)(PdsType*);

  /// Returns the Python type
  static PyTypeObject* typeObject();

  /// Builds Python object from corresponding Pds type, parent is the owner
  /// of the corresponding buffer space, usually XTC object. If destructor
  /// function is provided it will be called to delete the Pds object.
  static ConcreteType* PyObject_FromPds( PdsType* obj, PyObject* parent, size_t size, destructor dtor=0 );

  /// Builds Python object from the content of Xtc, parent is the owner
  /// of the corresponding buffer space, usually XTC object. If destructor
  /// function is provided it will be called to delete the Pds object.
  static ConcreteType* PyObject_FromXtc( const Pds::Xtc& xtc, PyObject* parent, destructor dtor=0 ) {
    return PyObject_FromPds( static_cast<PdsType*>((void*)xtc.payload()), parent, xtc.sizeofPayload(), dtor );
  }

  // returns pointer to an PdsType object
  static PdsType* pdsObject(PyObject* self) {
    PdsDataType* py_this = (PdsDataType*) self;
    if( ! py_this->m_obj ){
      PyErr_SetString(PyExc_TypeError, "Error: No Valid C++ Object");
    }
    return py_this->m_obj;
  }

  // returns true if object is an instance of this type or subtype
  static bool Object_TypeCheck( PyObject* obj ) {
    PyTypeObject* type = typeObject();
    return PyObject_TypeCheck( obj, type );
  }

  // Dump object info to a stream
  void print(std::ostream& out) const {
    out << "<" << ob_type->tp_name << "(@" << this << ")>";
  }
  
  // --------------------------------------------------

  PdsType* m_obj;
  PyObject* m_parent;
  size_t m_size;
  destructor m_dtor;

protected:

  /// Initialize Python type and register it in a module
  static void initType( const char* name, PyObject* module );

  // standard Python deallocation function
  static void PdsDataType_dealloc( PyObject* self );

#if PY_VERSION_HEX >= 0x02050000
  typedef Py_ssize_t PySsizeType;
#else
  typedef int PySsizeType;
#endif

  // class supports buffer interface
  static PySsizeType readbufferproc(PyObject* self, PySsizeType segment, void** ptrptr);
  static PySsizeType segcountproc(PyObject* self, PySsizeType* lenp);

  // repr() function
  static PyObject* repr( PyObject *self )  {
    std::ostringstream str;
    static_cast<ConcreteType*>(self)->print(str);
    return PyString_FromString( str.str().c_str() );
  }
};

/// stream insertion operator
template <typename ConcreteType, typename PdsType>
std::ostream&
operator<<(std::ostream& out, const PdsDataType<ConcreteType, PdsType>& data) {
  data.print(out);
  return out;
}

/// Returns the Python type opbject
template <typename ConcreteType, typename PdsType>
PyTypeObject*
PdsDataType<ConcreteType, PdsType>::typeObject()
{
  static PyBufferProcs bufferprocs = {
    readbufferproc, // bf_getreadbuffer
    0,              // bf_getwritebuffer
    segcountproc,   // bf_getsegcount
    0               // bf_getcharbuffer
  } ;

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
    repr,                    /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_sequence*/
    0,                       /*tp_as_mapping*/
    0,                       /*tp_hash*/
    0,                       /*tp_call*/
    repr,                    /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    &bufferprocs,            /*tp_as_buffer*/
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
ConcreteType*
PdsDataType<ConcreteType, PdsType>::PyObject_FromPds( PdsType* obj, PyObject* parent, size_t size, destructor dtor )
{
  ConcreteType* ob = PyObject_New(ConcreteType,typeObject());
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create PdsDataType object." );
    return 0;
  }

  ob->m_obj = obj;
  ob->m_parent = parent ;
  Py_XINCREF(ob->m_parent);
  ob->m_size = size;
  ob->m_dtor = dtor;

  return ob;
}

/// Initialize Python type and register it in a module
template <typename ConcreteType, typename PdsType>
void
PdsDataType<ConcreteType, PdsType>::initType( const char* name, PyObject* module )
{
  static std::string typeName;

  // perfix type name with module name
  const char* modname = PyModule_GetName(module);
  if ( modname ) {
    typeName = modname;
    typeName += '.';
  }
  typeName += name;

  // set the name
  PyTypeObject* type = typeObject();
  type->tp_name = (char*)typeName.c_str();

  // initialize type
  if ( PyType_Ready( type ) < 0 ) return;

  // register it in a module
  PyDict_SetItemString( PyModule_GetDict(module), (char*)name, (PyObject*) type );
}

template <typename ConcreteType, typename PdsType>
typename PdsDataType<ConcreteType, PdsType>::PySsizeType
PdsDataType<ConcreteType, PdsType>::readbufferproc(PyObject* self, PySsizeType segment, void** ptrptr)
{
  ConcreteType* py_this = static_cast<ConcreteType*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  *ptrptr = py_this->m_obj;
  return py_this->m_size;
}

template <typename ConcreteType, typename PdsType>
typename PdsDataType<ConcreteType, PdsType>::PySsizeType
PdsDataType<ConcreteType, PdsType>::segcountproc(PyObject* self, PySsizeType* lenp)
{
  ConcreteType* py_this = static_cast<ConcreteType*>(self);
  if( ! py_this->m_obj ){
    if ( lenp ) *lenp = 0;
    return 0;
  }

  if ( lenp ) {
    *lenp = py_this->m_size;
  }
  return 1;
}

} // namespace pypdsdata

#endif // PYPDSDATA_PDSDATATYPE_H
