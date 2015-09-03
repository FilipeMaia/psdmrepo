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
#include "LusiTime/Exceptions.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <cerrno>
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace LusiTime {

//----------------
// Constructors --
//----------------
Exception::Exception( const std::string& what )
  : std::runtime_error( "LusiTime::Exception: " + what )
{
}

ExceptionErrno::ExceptionErrno ( const std::string& what )
  : Exception( what + ": " + strerror(errno) )
{
}

ParseException::ParseException ( const std::string& what )
  : Exception( "date/time parse error: " + what )
{
}

} // namespace LusiTime
