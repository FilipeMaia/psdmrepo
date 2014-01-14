//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::L3T::ConfigV1, module_id)
  FUN0_WRAPPER(pypdsdata::L3T::ConfigV1, desc)

  PyMethodDef methods[] = {
    {"module_id",        module_id,       METH_NOARGS,  "self.module_id() -> string\n\nReturns module identification string." },
    {"desc",             desc,            METH_NOARGS,  "self.desc() -> string\n\nReturns description string." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::L3T::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::L3T::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::L3T::ConfigV1::print(std::ostream& str) const
{
  str << "L3T.ConfigV1(module_id=\"" << m_obj->module_id()
      << "\", desc=\"" << m_obj->desc()
      << "\")";
}
