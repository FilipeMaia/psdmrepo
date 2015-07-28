//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Exceptions...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "RootHist/Exceptions.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RootHist {

Exception::Exception(const ErrSvc::Context& ctx, const std::string& what)
  : ErrSvc::Issue(ctx, "RootHist::Exception: " + what)
{
}

ExceptionFileOpen::ExceptionFileOpen(const ErrSvc::Context& ctx, const std::string& name)
  : Exception(ctx, "failed to open ROOT file: " + name)
{  
}

} // namespace RootHist
