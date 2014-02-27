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
#include "PSXtcInput/Exceptions.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <cerrno>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcInput {

//----------------
// Constructors --
//----------------
Exception::Exception (const ErrSvc::Context& ctx, 
                      const std::string& className, 
                      const std::string& what)
  : ErrSvc::Issue(ctx, className+": "+what)
{
}

ExceptionErrno::ExceptionErrno ( const ErrSvc::Context& ctx, const std::string& what )
  : Exception( ctx, "ExceptionErrno", what + ": " + std::strerror(errno) )
{
}

} // namespace PSXtcInput
