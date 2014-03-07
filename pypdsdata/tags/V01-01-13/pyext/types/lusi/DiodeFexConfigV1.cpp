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

  // methods
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::DiodeFexConfigV1, base)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::DiodeFexConfigV1, scale)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"base",       base,   0, "List of NRANGES floating numbers", 0},
    {"scale",      scale,  0, "List of NRANGES floating numbers", 0},
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

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::Lusi::DiodeFexConfigV1::NRANGES);
  PyDict_SetItemString( type->tp_dict, "NRANGES", val );
  Py_XDECREF(val);

  BaseType::initType( "DiodeFexConfigV1", module );
}

void
pypdsdata::Lusi::DiodeFexConfigV1::print(std::ostream& str) const
{
  str << "lusi.DiodeFexConfigV1(base=" << m_obj->base()
      << ", scale=" << m_obj->scale()
      << ")" ;
}
