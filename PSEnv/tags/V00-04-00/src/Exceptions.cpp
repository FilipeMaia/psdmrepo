//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Exceptions...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSEnv/Exceptions.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <cxxabi.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  
  std::string typeName(const std::type_info& typeinfo)
  {
    int status;
    char* realname = abi::__cxa_demangle(typeinfo.name(), 0, 0, &status);
    std::string name = realname;
    free(realname);
    return name;
  }
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSEnv {

Exception::Exception( const ErrSvc::Context& ctx, const std::string& what )
  : ErrSvc::Issue( ctx, "PSEnv::Exception: " + what )
{
}

ExceptionEpicsName::ExceptionEpicsName ( const ErrSvc::Context& ctx, 
                                         const std::string& pvname ) 
  : Exception( ctx, "unknown EPICS PV name: " + pvname)
{  
}

ExceptionEpicsConversion::ExceptionEpicsConversion ( const ErrSvc::Context& ctx, 
                                                     const std::string& pvname,
                                                     const std::type_info& ti,
                                                     const std::string& what)
  : Exception( ctx, "error converting PV value: PV=" + pvname +
      ", result type=" + ::typeName(ti) + ", error: " + what )
{
}

} // namespace PSEnv
