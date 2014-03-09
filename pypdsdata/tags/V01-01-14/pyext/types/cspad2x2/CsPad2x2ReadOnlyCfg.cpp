//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ReadOnlyCfg...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CsPad2x2ReadOnlyCfg.h"

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
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::CsPad2x2::CsPad2x2ReadOnlyCfg, shiftTest)
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::CsPad2x2::CsPad2x2ReadOnlyCfg, version)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"shiftTest",       shiftTest,       0, "Integer number", 0},
    {"version",         version,         0, "Integer number", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad2x2::CsPad2x2ReadOnlyCfg class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad2x2::CsPad2x2ReadOnlyCfg::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "CsPad2x2ReadOnlyCfg", module );
}

void
pypdsdata::CsPad2x2::CsPad2x2ReadOnlyCfg::print(std::ostream& str) const
{
  str << "cspad2x2.CsPad2x2ReadOnlyCfg(shiftTest=" << m_obj.shiftTest()
      << ", version=" << m_obj.version()
      << ")";
}
