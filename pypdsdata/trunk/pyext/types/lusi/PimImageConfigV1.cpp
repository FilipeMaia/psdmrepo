//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: PimImageConfigV1.cpp 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//	Class PimImageConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PimImageConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

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
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::PimImageConfigV1, xscale)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Lusi::PimImageConfigV1, yscale)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"xscale",       xscale,   0, "Floating point number", 0},
    {"yscale",       yscale,   0, "Floating point number", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Lusi::PimImageConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Lusi::PimImageConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "PimImageConfigV1", module );
}

void
pypdsdata::Lusi::PimImageConfigV1::print(std::ostream& str) const
{
  str << "lusi.PimImageConfigV1(xscale=" << m_obj->xscale() << ", yscale=" << m_obj->yscale() << ")";
}
