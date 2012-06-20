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
 *  Exception classes
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace XtcInput {

class Exception : public std::runtime_error {
public:

  // Constructor
  Exception ( const std::string& className, const std::string& what ) ;

};

class ErrnoException : public Exception {
public:

    ErrnoException(const std::string& className, const std::string& what)
    : Exception( className, what + ": " + strerror(errno) ) {}

};


// thrown for incorrect arguments provided
class ArgumentException : public Exception {
public:

  ArgumentException( const std::string& msg )
    : Exception( "ArgumentException", msg ) {}

};

class FileOpenException : public ErrnoException {
public:

  FileOpenException( const std::string& fileName )
    : ErrnoException( "FileOpenException", "failed to open file "+fileName ) {}

};

class XTCEOFException : public Exception {
public:

    XTCEOFException(const std::string& fileName)
    : Exception( "XTCEOFException", "EOF while reading datagram payload in file "+fileName ) {}

};

class XTCReadException : public ErrnoException {
public:

  XTCReadException(const std::string& fileName)
    : ErrnoException( "XTCReadException", "failed to read XTC file "+fileName ) {}

};

class XTCSizeLimitException : public Exception {
public:

    XTCSizeLimitException(const std::string& fileName, size_t dgSize, size_t maxSize)
    : Exception( "XTCSizeLimitException", "datagram too large reading XTC file "+fileName+
            ": datagram size="+boost::lexical_cast<std::string>(dgSize) +
            ", max size="+boost::lexical_cast<std::string>(maxSize) ) {}

};

/// Generic XTC exception, just give it a message
class XTCGenException : public Exception {
public:

  XTCGenException( const std::string& msg )
    : Exception( "XTCGenException", msg ) {}

};

/// Error generated when merge mode string is invalid
class InvalidMergeMode : public Exception {
public:

  InvalidMergeMode( const std::string& str )
    : Exception( "InvalidMergeMode", "invalid merge mode string: " + str ) {}

};

} // namespace XtcInput

#endif // XTCINPUT_EXCEPTIONS_H
