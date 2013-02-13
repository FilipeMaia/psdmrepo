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
#include <map>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {
  
  // set of instances
  typedef std::map<ConfigSvc::ConfigSvc::context_t, boost::shared_ptr<ConfigSvc::ConfigSvcImplI> > ContextMap;
  ContextMap g_impl;
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ConfigSvc {

void
ConfigSvc::init(const boost::shared_ptr<ConfigSvcImplI>& impl, context_t context)
{
  if (g_impl.find(context) != g_impl.end()) throw ExceptionInitialized();
  g_impl.insert(std::make_pair(context, impl));
}

bool
ConfigSvc::initialized(context_t context)
{
  return g_impl.find(context) != g_impl.end();
}

ConfigSvcImplI&
ConfigSvc::impl(context_t context)
{
  ContextMap::const_iterator it = g_impl.find(context);
  if (it == g_impl.end()) throw ExceptionNotInitialized();
  return *it->second;
}

} // namespace ConfigSvc
