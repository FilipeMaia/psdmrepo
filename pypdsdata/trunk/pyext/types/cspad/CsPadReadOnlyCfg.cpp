//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadReadOnlyCfg...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CsPadReadOnlyCfg.h"

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
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::CsPad::CsPadReadOnlyCfg, shiftTest)
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::CsPad::CsPadReadOnlyCfg, version)
  
  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"shiftTest",       shiftTest,       0, "Integer number", 0},
    {"version",         version,         0, "Integer number", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad::CsPadReadOnlyCfg class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad::CsPadReadOnlyCfg::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "CsPadReadOnlyCfg", module );
}

