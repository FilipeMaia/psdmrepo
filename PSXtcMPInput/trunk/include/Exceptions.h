#ifndef PSXTCMPINPUT_EXCEPTIONS_H
#define PSXTCMPINPUT_EXCEPTIONS_H

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

namespace PSXtcMPInput {

/**
 *  Exception classes
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
  Exception ( const ErrSvc::Context& ctx, 
              const std::string& className, 
              const std::string& what ) ;

};

// thrown for missing worker info
class GenericException : public Exception {
public:

  GenericException(const ErrSvc::Context& ctx, const std::string& what)
    : Exception(ctx, "Exception", what) {}

};

// thrown for missing worker info
class MissingWorkers : public Exception {
public:

  MissingWorkers(const ErrSvc::Context& ctx)
    : Exception( ctx, "MissingWorkers", "Worker process information is missing from environment" ) {}

};

// thrown for empty file
class EmptyInput : public Exception {
public:

  EmptyInput(const ErrSvc::Context& ctx)
    : Exception( ctx, "EmptyInput", "XTC file(s) is empty" ) {}

};

// thrown for unexpected input
class UnexpectedInput : public Exception {
public:

  UnexpectedInput(const ErrSvc::Context& ctx)
    : Exception( ctx, "UnexpectedInput", "Number of datagrams received from source is not expected" ) {}

};

/// Exception class which extracts error info from errno.
class ExceptionErrno : public Exception {
public:

  /// Constructor takes the reason for an exception
  ExceptionErrno ( const ErrSvc::Context& ctx, const std::string& what ) ;

};

// thrown for missing worker info
class ProtocolError : public Exception {
public:

  ProtocolError(const ErrSvc::Context& ctx, const std::string& what)
    : Exception(ctx, "ProtocolError", what) {}

};

} // namespace PSXtcMPInput

#endif // PSXTCMPINPUT_EXCEPTIONS_H
