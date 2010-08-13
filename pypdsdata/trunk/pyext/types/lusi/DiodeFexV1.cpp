//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: DiodeFexV1.cpp 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//	Class DiodeFexV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DiodeFexV1.h"

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
  MEMBER_WRAPPER(pypdsdata::Lusi::DiodeFexV1, value)

  PyGetSetDef getset[] = {
    {"value",       value,   0, "", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Lusi::DiodeFexV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Lusi::DiodeFexV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = __repr__;
  type->tp_repr = __repr__;

  BaseType::initType( "DiodeFexV1", module );
}

namespace {

PyObject*
__repr__( PyObject *self )
{
  pypdsdata::Lusi::DiodeFexV1* py_this = (pypdsdata::Lusi::DiodeFexV1*) self;

  char buf[64];
  snprintf( buf, sizeof buf, "Lusi.DiodeFexV1(value=%g)", py_this->m_obj->value );
  return PyString_FromString( buf );
}
 
}
