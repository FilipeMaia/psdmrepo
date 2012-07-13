#ifndef PSXTCOUTPUT_EXCEPTIONS_H
#define PSXTCOUTPUT_EXCEPTIONS_H

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
#include <cerrno>
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

/// @addtogroup PSXtcOutput

/**
 *  @ingroup PSXtcOutput
 *
 *  Exception classes
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace PSXtcOutput {

class Exception : public ErrSvc::Issue {
public:

  // Constructor
  Exception ( const ErrSvc::Context& ctx,
              const std::string& className,
              const std::string& what ) ;

};

// thrown for file name formatiing errors
class FileNameFormatError : public Exception {
public:

  FileNameFormatError(const ErrSvc::Context& ctx, const std::string& fmt, const std::string& reason)
    : Exception( ctx, "FileNameFormatError", "Incorrect file name format string \"" + fmt + "\": " + reason) {}

};

// thrown for file open error
class FileOpenError : public Exception {
public:

  FileOpenError(const ErrSvc::Context& ctx, const std::string& fname)
    : Exception( ctx, "FileOpenError", "File open error for file \"" + fname + "\": " + std::strerror(errno)) {}

};

// thrown for write open error
class FileWriteError : public Exception {
public:

  FileWriteError(const ErrSvc::Context& ctx)
    : Exception( ctx, "FileOpenError", "Write error: " + std::string(std::strerror(errno))) {}

};

} // namespace PSXtcOutput

#endif // PSXTCOUTPUT_EXCEPTIONS_H
