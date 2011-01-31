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
#include "PsEvt/Exceptions.h"

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

namespace PsEvt {

Exception::Exception( const std::string& what )
  : std::runtime_error( "PsEvt::Exception: " + what )
{
}

ExceptionDuplicateKey::ExceptionDuplicateKey ( const std::type_info* typeinfo, 
                                               const Pds::DetInfo& detInfo, 
                                               const std::string& key ) 
  : Exception( "duplicate key: " + std::string(typeinfo->name()) + ":" + 
               Pds::DetInfo::name(detInfo) + ":" + key)
{  
}

} // namespace PsEvt
