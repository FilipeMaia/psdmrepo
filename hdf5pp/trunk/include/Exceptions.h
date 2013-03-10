#ifndef HDF5PP_EXCEPTIONS_H
#define HDF5PP_EXCEPTIONS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Exceptions.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/lexical_cast.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "ErrSvc/Issue.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace hdf5pp {

/// @addtogroup hdf5pp

/**
 *  @ingroup hdf5pp
 *
 *  Base class for all exceptions in the package
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */


class Exception : public ErrSvc::Issue {
public:

  // Constructor
  Exception ( const ErrSvc::Context& ctx, const std::string& className, const std::string& what )
    : ErrSvc::Issue( ctx, className+": "+what) {}

};

class Hdf5CallException : public Exception {
public:

  Hdf5CallException( const ErrSvc::Context& ctx, const std::string& function )
    : Exception( ctx, "Hdf5CallException", "HDF5 error in call to function "+function ) {}

};

class Hdf5DataSpaceSizeException : public Exception {
public:

  Hdf5DataSpaceSizeException( const ErrSvc::Context& ctx )
    : Exception( ctx, "Hdf5DataSpaceSizeException", "dataspace size mismatch" ) {}

};

class Hdf5BadTypeCast : public Exception {
public:

  Hdf5BadTypeCast( const ErrSvc::Context& ctx, const std::string& newtype )
    : Exception( ctx, "Hdf5BadTypeCast", "Illegal type cast to " + newtype ) {}

};

class Hdf5RankTooHigh : public Exception {
public:

  Hdf5RankTooHigh( const ErrSvc::Context& ctx, unsigned rank, unsigned maxrank)
    : Exception( ctx, "Hdf5RankTooHight", "Data rank (" + boost::lexical_cast<std::string>(rank) +
        ") is higher than max. supported rank (" + boost::lexical_cast<std::string>(maxrank) + ")") {}

};

class Hdf5RankMismatch : public Exception {
public:

  Hdf5RankMismatch( const ErrSvc::Context& ctx, unsigned expected, unsigned actual)
    : Exception( ctx, "Hdf5RankMismatch", "Data rank mismatch, expected rank "
        + boost::lexical_cast<std::string>(expected) + ", actual rank "
        + boost::lexical_cast<std::string>(actual)) {}

};


} // namespace hdf5pp

#endif // HDF5PP_EXCEPTIONS_H
