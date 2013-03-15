#ifndef PSHDF5INPUT_EXCEPTIONS_H
#define PSHDF5INPUT_EXCEPTIONS_H

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

/// @addgroup PSHdf5Input

namespace PSHdf5Input {

/**
 *  @brief Base class for exceptions generated in PSHdf5Input package.
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Exception : public ErrSvc::Issue {
public:

  // Constructor
  Exception ( const ErrSvc::Context& ctx, 
              const std::string& className, 
              const std::string& what ) ;

};

/// Exception thrown for empty file list
class NotHdf5Dataset : public Exception {
public:

  NotHdf5Dataset(const ErrSvc::Context& ctx, const std::string& ds)
    : Exception( ctx, "NotHdf5Dataset", "Input dataset is not an HDF5 data: " + ds ) {}

};

/// Exception thrown for empty file list
class EmptyFileList : public Exception {
public:

  EmptyFileList(const ErrSvc::Context& ctx)
    : Exception( ctx, "EmptyFileList", "No input file names specified" ) {}

};

/// Exception thrown when file cannot be open
class FileOpenError : public Exception {
public:

  FileOpenError(const ErrSvc::Context& ctx, const std::string& fileName, const std::string& reason)
    : Exception( ctx, "FileOpenError", "failed to open file '" + fileName +
        "': " + reason) {}

};

/// Exception thrown when data structure in file is bad
class FileStructure : public Exception {
public:

  FileStructure(const ErrSvc::Context& ctx, const std::string& reason)
    : Exception( ctx, "FileStructure", "HDF5 file structure error: " + reason) {}

};

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_EXCEPTIONS_H
