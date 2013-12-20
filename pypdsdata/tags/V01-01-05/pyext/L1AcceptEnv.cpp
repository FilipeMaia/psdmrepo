//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class L1AcceptEnv...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1AcceptEnv.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "EnumType.h"
#include "types/TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // helper method to get enum name out of enums array
  const char* enum_name(pypdsdata::EnumType::Enum enums[], int val, int maxval)
  {
    // try optimization first based on the order of enums in the list above
    if ( val <= maxval and enums[val].value == val ) {
      return enums[val].name;
    }
    // otherwise try linear search
    for( unsigned i = 0 ; enums[i].name ; ++ i ) {
      if (enums[i].value == val) return enums[i].name;
    }
    return "<Invalid>";    
  }

  pypdsdata::EnumType::Enum l3tResultEnumValues[] = {
      { "None_", Pds::L1AcceptEnv::None },
      { "Pass",  Pds::L1AcceptEnv::Pass },
      { "Fail",  Pds::L1AcceptEnv::Fail },
      { 0, 0 }
  };
  pypdsdata::EnumType l3tResultEnum ( "L3TResult", l3tResultEnumValues );
  
  inline const char* l3t_name(Pds::L1AcceptEnv::L3TResult res) {
    return enum_name(l3tResultEnumValues, int(res), 3);
  }
  
  // type-specific methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::L1AcceptEnv, value);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::L1AcceptEnv, clientGroupMask);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::L1AcceptEnv, trimmed);
  ENUM_FUN0_WRAPPER_EMBEDDED(pypdsdata::L1AcceptEnv, l3t_result, l3tResultEnum);

  PyMethodDef methods[] = {
    { "value",            value,            METH_NOARGS, "self.value() -> int\n\nReturns integer value." },
    { "clientGroupMask",  clientGroupMask,  METH_NOARGS, "self.clientGroupMask() -> int\n\nReturns integerbit mask value." },
    { "l3t_result",       l3t_result,       METH_NOARGS, "self.l3t_result() -> enum\n\nReturns enum value of type L3TResult." },
    { "trimmed",          trimmed,          METH_NOARGS, "self.trimmed() -> bool\n\nReturns true if the datagram content was trimmed." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::L1AcceptEnv class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::L1AcceptEnv::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "L3TResult", l3tResultEnum.type() );

  BaseType::initType( "L1AcceptEnv", module );
}

void 
pypdsdata::L1AcceptEnv::print(std::ostream& out) const
{
  out << "L1AcceptEnv(clientGroupMask=" << std::hex << std::showbase << m_obj.clientGroupMask() << std::dec 
      << ", l3t_result=" << ::l3t_name(m_obj.l3t_result()) << ")";
}
