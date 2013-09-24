//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadProtectionSystemThreshold...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CsPadProtectionSystemThreshold.h"

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
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::CsPad::CsPadProtectionSystemThreshold, adcThreshold)
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::CsPad::CsPadProtectionSystemThreshold, pixelCountThreshold)
  
  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"adcThreshold",        adcThreshold,        0, "Integer number", 0},
    {"pixelCountThreshold", pixelCountThreshold, 0, "Integer number", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad::ProtectionSystemThreshold class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad::CsPadProtectionSystemThreshold::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "CsPadProtectionSystemThreshold", module );
}
