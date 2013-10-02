//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Sample...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Sample.h"

//-----------------
// C/C++ Headers --
//-----------------

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

  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "channelsPerDevice",      Pds::Imp::Sample::channelsPerDevice },
        { 0, 0 }
  };

  // type-specific methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Imp::Sample, channels)

  PyMethodDef methods[] = {
    { "channels",       channels,       METH_NOARGS,
        "self.channels() -> list of int\n\nReturns list of integer numbers, one value per channel." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Imp::Sample class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Imp::Sample::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  pypdsdata::TypeLib::DefineEnums( type->tp_dict, ::enums );

  BaseType::initType( "Sample", module );
}

void
pypdsdata::Imp::Sample::print(std::ostream& out) const
{
  out << "imp.Sample(" << m_obj.channels() << ")";
}
