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

  // methods
  MEMBER_WRAPPER(pypdsdata::Lusi::IpmFexV1, sum)
  MEMBER_WRAPPER(pypdsdata::Lusi::IpmFexV1, xpos)
  MEMBER_WRAPPER(pypdsdata::Lusi::IpmFexV1, ypos)
  PyObject* IpmFexV1_channel( PyObject* self, void* );

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"sum",     sum,               0, "Floating point number", 0},
    {"xpos",    xpos,              0, "Floating point number", 0},
    {"ypos",    ypos,              0, "Floating point number", 0},
    {"channel", IpmFexV1_channel,  0, "List of 4 floating numbers", 0},
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

  BaseType::initType( "IpmFexV1", module );
}

void
pypdsdata::Lusi::IpmFexV1::print(std::ostream& str) const
{
  str << "lusi.IpmFexV1(sum=" << m_obj->sum
      << ", xpos=" << m_obj->xpos
      << ", ypos=" << m_obj->ypos
      << ", channel=[" ;

  const int size = sizeof m_obj->channel / sizeof m_obj->channel[0];
  for ( int i = 0 ; i < size ; ++ i ) {
    if ( i ) str << ", " ;
    str << m_obj->channel[i];
  }
  str << "])" ;
}


namespace {

PyObject*
IpmFexV1_channel( PyObject* self, void* )
{
  Pds::Lusi::IpmFexV1* obj = pypdsdata::Lusi::IpmFexV1::pdsObject(self);
  if (not obj) return 0;

  const int size = sizeof obj->channel / sizeof obj->channel[0];
  PyObject* list = PyList_New( size );
  for ( int i = 0 ; i < size ; ++ i ) {
    PyList_SET_ITEM( list, i, pypdsdata::TypeLib::toPython(obj->channel[i]) );
  }
  return list;
}

}
