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
  MEMBER_WRAPPER(pypdsdata::CsPad2x2::CsPad2x2ReadOnlyCfg, shiftTest)
  MEMBER_WRAPPER(pypdsdata::CsPad2x2::CsPad2x2ReadOnlyCfg, version)
  PyObject* _repr( PyObject *self );

  PyGetSetDef getset[] = {
    {"shiftTest",       shiftTest,       0, "", 0},
    {"version",         version,         0, "", 0},
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "CsPad2x2ReadOnlyCfg", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::CsPad2x2::CsPad2x2ReadOnlyCfg* pdsObj = pypdsdata::CsPad2x2::CsPad2x2ReadOnlyCfg::pdsObject(self);
  if(not pdsObj) return 0;

  std::ostringstream str;
  str << "cspad2x2.CsPad2x2ReadOnlyCfg(shiftTest=" << pdsObj->shiftTest
      << ", version=" << pdsObj->version
      << ")";
  return PyString_FromString( str.str().c_str() );
}

}
