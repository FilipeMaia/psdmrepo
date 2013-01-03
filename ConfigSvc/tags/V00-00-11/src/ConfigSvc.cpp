//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigSvc...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigSvc/ConfigSvc.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  
  // one global configuration service instance
  std::auto_ptr<ConfigSvc::ConfigSvcImplI> g_impl;
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ConfigSvc {

void
ConfigSvc::init(std::auto_ptr<ConfigSvcImplI> impl)
{
  if (g_impl.get()) throw ExceptionInitialized();
  g_impl = impl;
}

ConfigSvcImplI&
ConfigSvc::impl()
{
  ConfigSvcImplI* p = g_impl.get();
  if (not p) throw ExceptionNotInitialized();
  return *p;
}

} // namespace ConfigSvc
