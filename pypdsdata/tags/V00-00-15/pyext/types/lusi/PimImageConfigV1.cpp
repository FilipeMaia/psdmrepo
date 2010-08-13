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
  PyObject* __repr__( PyObject *self );

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
  type->tp_str = __repr__;
  type->tp_repr = __repr__;

  BaseType::initType( "PimImageConfigV1", module );
}

namespace {

PyObject*
__repr__( PyObject *self )
{
  pypdsdata::Lusi::PimImageConfigV1* py_this = (pypdsdata::Lusi::PimImageConfigV1*) self;

  char buf[64];
  snprintf( buf, sizeof buf, "Lusi.PimImageConfigV1(xscale=%g, yscale=%g)", 
      py_this->m_obj->xscale, py_this->m_obj->yscale );
  return PyString_FromString( buf );
}
 
}
