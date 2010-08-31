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
#include <stdexcept>

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
class Exception : public std::runtime_error {
public:

  // Constructor
  Exception ( const std::string& className, const std::string& what )
    : std::runtime_error( className+": "+what) {}

};

class Hdf5CallException : public Exception {
public:

  Hdf5CallException( const std::string& method, const std::string& function )
    : Exception( "Hdf5CallException", method+" - HDF5 error in call to function "+function ) {}

};

class Hdf5DataSpaceSizeException : public Exception {
public:

  Hdf5DataSpaceSizeException( const std::string& method )
    : Exception( "Hdf5DataSpaceSizeException", method+" - dataspace size mismatch" ) {}

};


} // namespace hdf5pp

#endif // HDF5PP_EXCEPTIONS_H
