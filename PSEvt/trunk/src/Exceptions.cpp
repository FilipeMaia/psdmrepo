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
                                               const std::type_info* typeinfo, 
                                               const Pds::Src& source, 
                                               const std::string& key ) 
  : Exception( ctx, "duplicate key: " + std::string(typeinfo->name()) + ":phy(" + 
      boost::lexical_cast<std::string>(source.phy()) + "):" + key)
{  
}

ExceptionSourceFormat::ExceptionSourceFormat ( const ErrSvc::Context& ctx, 
                                               const std::string& format ) 
  : Exception( ctx, "Source string cannot be parsed: '" + format + "'")
{  
}

} // namespace PSEvt
