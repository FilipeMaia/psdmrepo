#ifndef PSTIME_EXCEPTIONS_H
#define PSTIME_EXCEPTIONS_H

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

namespace PSTime {

/**
 *  @brief Base class for exceptions in PSTime package
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @version \$Id$
 *
 *  @author Andrei Salnikov
 */

/// Base class for all exception classes in a package
class Exception : public ErrSvc::Issue {
public:

  // Constructor
  Exception ( const ErrSvc::Context& ctx, 
              const std::string& className, 
              const std::string& what ) 
    : ErrSvc::Issue(ctx, className+": "+what) 
  {}

};

/// Exception generated when standard library produces an error with corresponding errno.
class ErrnoException : public Exception {
public:

    ErrnoException(const ErrSvc::Context& ctx, 
                   const std::string& what)
    : Exception( ctx, "ErrnoException", what + ": " + strerror(errno) ) {}

};

/// Exception generated for failed clock_gettime() function.
class GetTimeException : public ErrnoException {
public:

  GetTimeException(const ErrSvc::Context& ctx)
    : ErrnoException( ctx, "clock_gettime failed") {}

};

/// Exception generated for failures during time string parsing.
class TimeParseException : public Exception {
public:

  TimeParseException(const ErrSvc::Context& ctx, const std::string& timestr)
    : Exception( ctx, "TimeParseException", "time parse failed: "+timestr) {}

};

} // namespace PSTime

#endif // PSTIME_EXCEPTIONS_H
