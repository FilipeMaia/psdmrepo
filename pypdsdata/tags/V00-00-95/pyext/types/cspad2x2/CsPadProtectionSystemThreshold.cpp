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
  MEMBER_WRAPPER(pypdsdata::CsPad2x2::CsPadProtectionSystemThreshold, adcThreshold)
  MEMBER_WRAPPER(pypdsdata::CsPad2x2::CsPadProtectionSystemThreshold, pixelCountThreshold)
  PyObject* _repr( PyObject *self );

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"adcThreshold",        adcThreshold,        0, "Integer number", 0},
    {"pixelCountThreshold", pixelCountThreshold, 0, "Integer number", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::CsPad2x2::ProtectionSystemThreshold class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::CsPad2x2::CsPadProtectionSystemThreshold::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "CsPadProtectionSystemThreshold", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::CsPad2x2::ProtectionSystemThreshold* pdsObj = pypdsdata::CsPad2x2::CsPadProtectionSystemThreshold::pdsObject(self);
  if(not pdsObj) return 0;

  std::ostringstream str;
  str << "cspad2x2.CsPadProtectionSystemThreshold(adcThreshold=" << pdsObj->adcThreshold
      << ", pixelCountThreshold=" << pdsObj->pixelCountThreshold
      << ")";
  return PyString_FromString( str.str().c_str() );
}

}
