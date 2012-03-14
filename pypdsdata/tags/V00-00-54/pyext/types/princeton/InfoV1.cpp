//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class InfoV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "InfoV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigV1.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Princeton::InfoV1, temperature)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "temperature",    temperature,    METH_NOARGS, "self.temperature() -> float\n\nReturns temperature value" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Princeton::InfoV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Princeton::InfoV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "InfoV1", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::Princeton::InfoV1* obj = pypdsdata::Princeton::InfoV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "princeton.InfoV1(temperature=" << obj->temperature() << ")";

  return PyString_FromString( str.str().c_str() );
}

}
