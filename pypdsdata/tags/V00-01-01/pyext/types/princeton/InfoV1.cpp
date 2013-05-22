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

  BaseType::initType( "InfoV1", module );
}

void
pypdsdata::Princeton::InfoV1::print(std::ostream& out) const
{
  out << "princeton.InfoV1(temperature=" << m_obj->temperature() << ")";
}
