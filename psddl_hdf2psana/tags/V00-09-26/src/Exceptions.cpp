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
#include "psddl_hdf2psana/Exceptions.h"

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

namespace psddl_hdf2psana {

Exception::Exception( const ErrSvc::Context& ctx, const std::string& what )
  : ErrSvc::Issue( ctx, "psddl_hdf2psana::Exception: " + what )
{
}

ExceptionGroupSourceName::ExceptionGroupSourceName ( const ErrSvc::Context& ctx,
                                                     const std::string& group )
  : Exception( ctx, "group name cannot be converted to source address: " + group)
{
}

ExceptionGroupTypeIdName::ExceptionGroupTypeIdName ( const ErrSvc::Context& ctx,
                                                     const std::string& group )
  : Exception( ctx, "group name cannot be converted to TypeId: " + group)
{
}

ExceptionDataRank::ExceptionDataRank( const ErrSvc::Context& ctx, int rank, int expectedRank )
  : Exception( ctx, "data rank mismatch, expected rank: " +
      boost::lexical_cast<std::string>(expectedRank) + ", real rank: " +
      boost::lexical_cast<std::string>(rank) )
{
}

ExceptionSchemaVersion::ExceptionSchemaVersion( const ErrSvc::Context& ctx, const std::string& type, int version )
  : Exception( ctx, "unknown schema version number: type " + type +
      ", version: " + boost::lexical_cast<std::string>(version) )
{
}

ExceptionNotImplemented::ExceptionNotImplemented( const ErrSvc::Context& ctx, const std::string& msg )
  : Exception( ctx, "not implemented: " + msg )
{
}

} // namespace psddl_hdf2psana
