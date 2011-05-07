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
#include "PSEvt/Exceptions.h"

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

namespace PSEvt {

Exception::Exception( const ErrSvc::Context& ctx, const std::string& what )
  : ErrSvc::Issue( ctx, "PSEvt::Exception: " + what )
{
}

ExceptionDuplicateKey::ExceptionDuplicateKey ( const ErrSvc::Context& ctx, 
                                               const EventKey& key ) 
  : Exception( ctx, "duplicate key: " + boost::lexical_cast<std::string>(key))
{  
}

ExceptionSourceFormat::ExceptionSourceFormat ( const ErrSvc::Context& ctx, 
                                               const std::string& format ) 
  : Exception( ctx, "Source string cannot be parsed: '" + format + "'")
{  
}

} // namespace PSEvt
