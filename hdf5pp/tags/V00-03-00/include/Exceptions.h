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

/**
 *  Exception classes used throughout the package.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace hdf5pp {

/// Base class for all exceptions in the package
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


} // namespace hdf5pp

#endif // HDF5PP_EXCEPTIONS_H
