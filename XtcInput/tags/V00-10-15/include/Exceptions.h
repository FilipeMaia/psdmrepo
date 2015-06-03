#ifndef XTCINPUT_EXCEPTIONS_H
#define XTCINPUT_EXCEPTIONS_H

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
#include <cerrno>
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

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Exception classes
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class Exception : public ErrSvc::Issue {
public:

  // Constructor
  Exception(const ErrSvc::Context& ctx, const std::string& className, const std::string& what);

};


class ErrnoException : public Exception {
public:

    ErrnoException(const ErrSvc::Context& ctx, const std::string& className, const std::string& what)
    : Exception(ctx, className, what + ": " + strerror(errno)) {}

};


// thrown for incorrect arguments provided
class ArgumentException : public Exception {
public:

  ArgumentException(const ErrSvc::Context& ctx, const std::string& msg)
    : Exception(ctx, "ArgumentException", msg) {}

};

class FileOpenException : public ErrnoException {
public:

  FileOpenException(const ErrSvc::Context& ctx, const std::string& fileName)
    : ErrnoException(ctx, "FileOpenException", "failed to open file "+fileName) {}

};

class FileNotInStream : public Exception {
public:

  FileNotInStream(const ErrSvc::Context& ctx, const std::string& fileName)
    : Exception(ctx, "FileNotInStream", "ChunkFileIter does not contain file: " +fileName) {}

};

class StreamNotInPosition : public Exception {
public:

  StreamNotInPosition(const ErrSvc::Context& ctx, const std::string & stream)
    : Exception(ctx, "StreamNotInPosition", "No file in Position for stream " + stream) {}
};

class JumpToDifferentRun : public Exception {
public:

  JumpToDifferentRun(const ErrSvc::Context& ctx) : Exception(ctx, "JumpToDifferentRun","") {}

};

class XTCLiveTimeout : public Exception {
public:

  XTCLiveTimeout(const ErrSvc::Context& ctx, const std::string& fileName, int timeout)
    : Exception(ctx, "XTCLiveTimeout", "Timeout (" + boost::lexical_cast<std::string>(timeout) +
        " seconds) while reading live data from file "+fileName) {}

};

class ErrDbRunLiveData : public Exception {
public:

  ErrDbRunLiveData(const ErrSvc::Context& ctx, int run)
    : Exception(ctx, "ErrDbRunLiveData", "No files in database for run #"+boost::lexical_cast<std::string>(run)) {}

};

class XTCReadException : public ErrnoException {
public:

  XTCReadException(const ErrSvc::Context& ctx, const std::string& fileName)
    : ErrnoException(ctx, "XTCReadException", "failed to read XTC file "+fileName ) {}

};

class XTCSizeLimitException : public Exception {
public:

    XTCSizeLimitException(const ErrSvc::Context& ctx, const std::string& fileName, size_t dgSize, size_t maxSize)
    : Exception(ctx, "XTCSizeLimitException", "datagram too large reading XTC file "+fileName+
            ": datagram size="+boost::lexical_cast<std::string>(dgSize) +
            ", max size="+boost::lexical_cast<std::string>(maxSize) ) {}

};

/// Generic XTC exception, just give it a message
class XTCGenException : public Exception {
public:

  XTCGenException(const ErrSvc::Context& ctx, const std::string& msg)
    : Exception(ctx, "XTCGenException", msg ) {}

};

/// Error generated when merge mode string is invalid
class InvalidMergeMode : public Exception {
public:

  InvalidMergeMode(const ErrSvc::Context& ctx, const std::string& str)
    : Exception(ctx, "InvalidMergeMode", "invalid merge mode string: " + str ) {}

};

/// Error generated when dataset specification is incorrect
class DatasetSpecError : public Exception {
public:

  DatasetSpecError(const ErrSvc::Context& ctx, const std::string& str)
    : Exception(ctx, "DatasetSpecError", "invalid dataset specification: " + str) {}

};

/// Error generated when dataset directory is not found
class DatasetDirError : public Exception {
public:

  DatasetDirError(const ErrSvc::Context& ctx, const std::string& str)
    : Exception(ctx, "DatasetDirError", "dataset directory is missing: " + str) {}

};

/// Error generated when multiple live directories are specified
class LiveDirError : public Exception {
public:

  LiveDirError(const ErrSvc::Context& ctx)
    : Exception(ctx, "LiveDirError", "only one live data directory is supported") {}

};

/// Error generated when incorrect or conflicting live streams are specified
class LiveStreamError : public Exception {
public:

  LiveStreamError(const ErrSvc::Context& ctx)
    : Exception(ctx, "LiveStreamError", "incorrect or conflicting live stream specified") {}

};

/// Exception thrown when dataset specification does not produce any files
class NoFilesInDataset : public Exception {
public:

    NoFilesInDataset(const ErrSvc::Context& ctx, const std::string& ds)
    : Exception( ctx, "NoFilesInDataset", "Dataset has no files, check dataset specification: '" + ds + "'") {}

};

/// XTC exception for invalid extent size
class XTCExtentException : public Exception {
public:

  XTCExtentException(const ErrSvc::Context& ctx, const std::string& fileName, size_t offset, uint32_t extent)
    : Exception(ctx, "XTCExtentException", "Invalid XTC extent size in file " + fileName +
        ": offset=" + boost::lexical_cast<std::string>(offset) +
        ", extent=" + boost::lexical_cast<std::string>(extent)) {}

};

} // namespace XtcInput

#endif // XTCINPUT_EXCEPTIONS_H
