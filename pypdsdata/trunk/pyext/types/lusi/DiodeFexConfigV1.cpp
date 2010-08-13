//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DiodeFexConfigV1.cpp 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//	Class DiodeFexConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DiodeFexConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  PyObject* __repr__( PyObject *self );

  // methods
  PyObject* DiodeFexConfigV1_base( PyObject* self, void* );
  PyObject* DiodeFexConfigV1_scale( PyObject* self, void* );

  PyGetSetDef getset[] = {
    {"base",       DiodeFexConfigV1_base,   0, "", 0},
    {"scale",      DiodeFexConfigV1_scale,  0, "", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Lusi::DiodeFexConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Lusi::DiodeFexConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = __repr__;
  type->tp_repr = __repr__;

  BaseType::initType( "DiodeFexConfigV1", module );
}


namespace {

PyObject*
DiodeFexConfigV1_base( PyObject* self, void* )
{
  pypdsdata::Lusi::DiodeFexConfigV1* py_this = static_cast<pypdsdata::Lusi::DiodeFexConfigV1*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  const int size = Pds::Lusi::DiodeFexConfigV1::NRANGES;
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    PyList_SET_ITEM( list, i, pypdsdata::TypeLib::toPython(py_this->m_obj->base[i]) );
  }
  return list;
}

PyObject*
DiodeFexConfigV1_scale( PyObject* self, void* )
{
  pypdsdata::Lusi::DiodeFexConfigV1* py_this = static_cast<pypdsdata::Lusi::DiodeFexConfigV1*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  const int size = Pds::Lusi::DiodeFexConfigV1::NRANGES;
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    PyList_SET_ITEM( list, i, pypdsdata::TypeLib::toPython(py_this->m_obj->scale[i]) );
  }
  return list;
}

PyObject*
__repr__( PyObject *self )
{
  pypdsdata::Lusi::DiodeFexConfigV1* py_this = (pypdsdata::Lusi::DiodeFexConfigV1*) self;

  std::ostringstream str ;
  str << "Lusi.DiodeFexConfigV1(base=[" ;
  
  const int size = Pds::Lusi::DiodeFexConfigV1::NRANGES;
  for ( int i = 0 ; i < size ; ++ i ) {
    if ( i ) str << ", " ;
    str << py_this->m_obj->base[i];
  }
  str << "], scale=[" ;
  for ( int i = 0 ; i < size ; ++ i ) {
    if ( i ) str << ", " ;
    str << py_this->m_obj->scale[i];
  }
  str << "])" ;

  return PyString_FromString( str.str().c_str() );
}

}
