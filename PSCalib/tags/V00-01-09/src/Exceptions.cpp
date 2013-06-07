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
#include "PSCalib/Exceptions.h"

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

namespace PSCalib {

Exception::Exception( const ErrSvc::Context& ctx, const std::string& what )
  : ErrSvc::Issue( ctx, "PSCalib::Exception: " + what )
{
}

NotDetInfoError::NotDetInfoError ( const ErrSvc::Context& ctx )
  : Exception(ctx, "Source address is not DetInfo address")
{
}

} // namespace PSCalib
