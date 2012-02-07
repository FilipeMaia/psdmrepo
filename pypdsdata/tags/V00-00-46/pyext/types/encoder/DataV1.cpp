//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // methods
  MEMBER_WRAPPER(pypdsdata::Encoder::DataV1, _33mhz_timestamp)
  MEMBER_WRAPPER(pypdsdata::Encoder::DataV1, _encoder_count)
  PyObject* _repr( PyObject *self );

  PyGetSetDef getset[] = {
    {"_33mhz_timestamp", _33mhz_timestamp,  0, "", 0},
    {"_encoder_count",   _encoder_count,    0, "", 0},
    {0, 0, 0, 0, 0}
  };

  // methods
  FUN0_WRAPPER(pypdsdata::Encoder::DataV1, value)

  PyMethodDef methods[] = {
    {"value",                 value,                  METH_NOARGS,  "" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Encoder::DataV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Encoder::DataV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "DataV1", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::Encoder::DataV1* obj = pypdsdata::Encoder::DataV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "encoder.DataV1(33mhz_timestamp=" << obj->_33mhz_timestamp
      << ", encoder_count=" << obj->_encoder_count
      << ")";
  
  return PyString_FromString( str.str().c_str() );
}

}
