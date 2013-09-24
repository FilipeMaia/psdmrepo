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
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::IpmFexV1, sum)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::IpmFexV1, xpos)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::IpmFexV1, ypos)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::IpmFexV1, channel)
  PyObject* IpmFexV1_channel( PyObject* self, void* );

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"sum",     sum,               0, "Floating point number", 0},
    {"xpos",    xpos,              0, "Floating point number", 0},
    {"ypos",    ypos,              0, "Floating point number", 0},
    {"channel", channel,           0, "List of 4 floating numbers", 0},
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
  str << "lusi.IpmFexV1(sum=" << m_obj->sum()
      << ", xpos=" << m_obj->xpos()
      << ", ypos=" << m_obj->ypos()
      << ", channel=" << m_obj->channel()
      << ")" ;
}
