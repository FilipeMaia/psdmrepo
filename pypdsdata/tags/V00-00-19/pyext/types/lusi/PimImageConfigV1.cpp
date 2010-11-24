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

  // standard Python stuff
  PyObject* _repr( PyObject *self );

  // methods
  MEMBER_WRAPPER(pypdsdata::Lusi::PimImageConfigV1, xscale)
  MEMBER_WRAPPER(pypdsdata::Lusi::PimImageConfigV1, yscale)

  PyGetSetDef getset[] = {
    {"xscale",       xscale,   0, "", 0},
    {"yscale",       yscale,   0, "", 0},
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "PimImageConfigV1", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::Lusi::PimImageConfigV1* obj = pypdsdata::Lusi::PimImageConfigV1::pdsObject(self);
  if (not obj) return 0;

  char buf[64];
  snprintf( buf, sizeof buf, "lusi.PimImageConfigV1(xscale=%g, yscale=%g)", 
      obj->xscale, obj->yscale );
  return PyString_FromString( buf );
}
 
}
