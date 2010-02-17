#ifndef PYPDSDATA_TYPELIB_H
#define PYPDSDATA_TYPELIB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TypeLib.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <Python.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PdsDataType.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {

/**
 *  Type traits library to simplify conversion between Pds types and Python types.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace TypeLib {

/// free-standing functions for conversion from C++ types to Python types
inline PyObject* toPython(bool v) { return PyBool_FromLong( v ); }
inline PyObject* toPython(int v) { return PyInt_FromLong( v ); }
inline PyObject* toPython(unsigned int v) { return PyInt_FromLong( v ); }
inline PyObject* toPython(long int v) { return PyInt_FromLong( v ); }
inline PyObject* toPython(unsigned long int v) { return PyLong_FromUnsignedLong( v ); }
inline PyObject* toPython(float v) { return PyFloat_FromDouble( v ); }
inline PyObject* toPython(double v) { return PyFloat_FromDouble( v ); }
inline PyObject* toPython(const char* v) { return PyString_FromString( v ); }

// traits
template <typename T>
struct TypeCvt {
};

template <>
struct TypeCvt<int> {
  static PyObject* toPython(int v) { return PyInt_FromLong( v ); }
};

template <>
struct TypeCvt<unsigned int> {
  static PyObject* toPython(unsigned int v) { return PyInt_FromLong( v ); }
};

template <>
struct TypeCvt<float> {
  static PyObject* toPython(float v) { return PyFloat_FromDouble( v ); }
};

template <>
struct TypeCvt<double> {
  static PyObject* toPython(double v) { return PyFloat_FromDouble( v ); }
};


#define FUN0_WRAPPER(PYTYPE,FUN0) \
  PyObject* FUN0( PyObject* self, PyObject* ) { \
    PYTYPE* py_this = (PYTYPE*) self; \
    if( ! py_this->m_obj ){ \
      PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object"); \
      return 0; \
    } \
    return pypdsdata::TypeLib::toPython( py_this->m_obj->FUN0() ); \
  }

#define FUN0_WRAPPER_EMBEDDED(PYTYPE,FUN0) \
  PyObject* FUN0( PyObject* self, PyObject* ) { \
    PYTYPE* py_this = (PYTYPE*) self; \
    return pypdsdata::TypeLib::toPython( py_this->m_obj.FUN0() ); \
  }

#define ENUM_FUN0_WRAPPER(PYTYPE,FUN0,ENUM) \
  PyObject* FUN0( PyObject* self, PyObject* ) { \
    PYTYPE* py_this = (PYTYPE*) self; \
    if( ! py_this->m_obj ){ \
      PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object"); \
      return 0; \
    } \
    return ENUM.Enum_FromLong( py_this->m_obj->FUN0() ); \
  }

#define ENUM_FUN0_WRAPPER_EMBEDDED(PYTYPE,FUN0,ENUM) \
  PyObject* FUN0( PyObject* self, PyObject* ) { \
    PYTYPE* py_this = (PYTYPE*) self; \
    return ENUM.Enum_FromLong( py_this->m_obj.FUN0() ); \
  }

#define MEMBER_WRAPPER(PYTYPE,MEMBER) \
  PyObject* MEMBER( PyObject* self, void* ) { \
    PYTYPE* py_this = (PYTYPE*) self; \
    if( ! py_this->m_obj ){ \
      PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object"); \
      return 0; \
    } \
    return pypdsdata::TypeLib::toPython( py_this->m_obj->MEMBER ); \
  }

#define MEMBER_WRAPPER_EMBEDDED(PYTYPE,MEMBER) \
  PyObject* MEMBER( PyObject* self, void* ) { \
    PYTYPE* py_this = (PYTYPE*) self; \
    return pypdsdata::TypeLib::toPython( py_this->m_obj.MEMBER ); \
  }


struct EnumEntry {
  const char* name ;
  int value;
};

inline void DefineEnums( PyObject* dict, const EnumEntry* enums )
{
  for ( ; enums->name ; ++ enums ) {
    PyObject* val = PyInt_FromLong(enums->value);
    PyDict_SetItemString( dict, enums->name, val );
    Py_DECREF( val );
  }
}


} // namespace TypeLib

} // namespace pypdsdata

#endif // PYPDSDATA_TYPELIB_H
