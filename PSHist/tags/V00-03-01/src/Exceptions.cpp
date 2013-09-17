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
#include "PSHist/Exceptions.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSHist {

Exception::Exception(const ErrSvc::Context& ctx, const std::string& what)
  : ErrSvc::Issue(ctx, "PSHist::Exception: " + what)
{
}

ExceptionDuplicateName::ExceptionDuplicateName(const ErrSvc::Context& ctx, const std::string& name)
  : Exception(ctx, "duplicate histogram/tuple name: " + name)
{  
}

ExceptionBins::ExceptionBins(const ErrSvc::Context& ctx)
  : Exception(ctx, "zero number of bins specified")
{  
}

ExceptionAxisRange::ExceptionAxisRange(const ErrSvc::Context& ctx, double xlow, double xhigh)
  : Exception(ctx, "invalid axis range: low=" + boost::lexical_cast<std::string>(xlow) 
      + " high=" + boost::lexical_cast<std::string>(xhigh))
{  
}

ExceptionAxisEdgeOrder::ExceptionAxisEdgeOrder(const ErrSvc::Context& ctx)
  : Exception(ctx, "axis edges have incorrect ordering")
{  
}

ExceptionStore::ExceptionStore(const ErrSvc::Context& ctx, const std::string& reason)
  : Exception(ctx, "failed to store histograms: " + reason)
{  
}

ExceptionDuplicateColumn::ExceptionDuplicateColumn(const ErrSvc::Context& ctx, 
    const std::string& tupleName, const std::string& columnName)
  : Exception(ctx, "duplicate tuple column name: " + tupleName + "." + columnName)
{  
}

} // namespace PSHist
