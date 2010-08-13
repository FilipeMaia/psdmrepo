//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: IpmFexV1.cpp 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//	Class IpmFexV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "IpmFexV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DiodeFexConfigV1.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  PyObject* __repr__( PyObject *self );

  // methods
  MEMBER_WRAPPER(pypdsdata::Lusi::IpmFexV1, sum)
  MEMBER_WRAPPER(pypdsdata::Lusi::IpmFexV1, xpos)
  MEMBER_WRAPPER(pypdsdata::Lusi::IpmFexV1, ypos)
  PyObject* IpmFexV1_channel( PyObject* self, void* );

  PyGetSetDef getset[] = {
    {"sum",     sum,               0, "", 0},
    {"xpos",    xpos,              0, "", 0},
    {"ypos",    ypos,              0, "", 0},
    {"channel", IpmFexV1_channel,  0, "", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Lusi::IpmFexV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Lusi::IpmFexV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = __repr__;
  type->tp_repr = __repr__;

  BaseType::initType( "IpmFexV1", module );
}


namespace {

PyObject*
IpmFexV1_channel( PyObject* self, void* )
{
  pypdsdata::Lusi::IpmFexV1* py_this = static_cast<pypdsdata::Lusi::IpmFexV1*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  const int size = sizeof py_this->m_obj->channel / sizeof py_this->m_obj->channel[0];
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    PyList_SET_ITEM( list, i, pypdsdata::TypeLib::toPython(py_this->m_obj->channel[i]) );
  }
  return list;
}

PyObject*
__repr__( PyObject *self )
{
  pypdsdata::Lusi::IpmFexV1* py_this = (pypdsdata::Lusi::IpmFexV1*) self;

  std::ostringstream str ;
  str << "Lusi.IpmFexV1(sum=" << py_this->m_obj->sum
      << ", xpos=" << py_this->m_obj->xpos 
      << ", ypos=" << py_this->m_obj->ypos 
      << ", channel=[" ;
  
  const int size = sizeof py_this->m_obj->channel / sizeof py_this->m_obj->channel[0];
  for ( int i = 0 ; i < size ; ++ i ) {
    if ( i ) str << ", " ;
    str << py_this->m_obj->channel[i];
  }
  str << "])" ;

  return PyString_FromString( str.str().c_str() );
}

}
