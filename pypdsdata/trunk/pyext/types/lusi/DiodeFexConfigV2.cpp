//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DiodeFexConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DiodeFexConfigV2.h"

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
MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::DiodeFexConfigV2, base)
MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::DiodeFexConfigV2, scale)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"base",       base,   0, "List of NRANGES floating numbers", 0},
    {"scale",      scale,  0, "List of NRANGES floating numbers", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Lusi::DiodeFexConfigV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Lusi::DiodeFexConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyObject* val = PyInt_FromLong(Pds::Lusi::DiodeFexConfigV2::NRANGES);
  PyDict_SetItemString( type->tp_dict, "NRANGES", val );
  Py_XDECREF(val);

  BaseType::initType( "DiodeFexConfigV2", module );
}

void
pypdsdata::Lusi::DiodeFexConfigV2::print(std::ostream& str) const
{
  str << "lusi.DiodeFexConfigV2(base=" << m_obj->base()
      << ", scale=" << m_obj->scale()
      << ")" ;
}
