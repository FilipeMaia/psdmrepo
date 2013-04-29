#ifndef PSSHMEMINPUT_EXCEPTIONS_H
#define PSSHMEMINPUT_EXCEPTIONS_H

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

namespace PSShmemInput {

class Exception : public ErrSvc::Issue {
public:

  // Constructor
  Exception ( const ErrSvc::Context& ctx, 
              const std::string& className, 
              const std::string& what )
    : ErrSvc::Issue(ctx, className+": "+what)
  {}

};

// thrown for empty input list
class EmptyInputList : public Exception {
public:

  EmptyInputList(const ErrSvc::Context& ctx)
    : Exception( ctx, "EmptyInputList", "No input source name specified" ) {}

};

// thrown for too long input list
class DatasetSpecError : public Exception {
public:

  DatasetSpecError(const ErrSvc::Context& ctx, const std::string& err, const std::string& spec)
    : Exception( ctx, "DatasetSpecError", "Dataset specification error: " + err + " (" + spec + ")" ) {}

};

} // namespace PSXtcInput

#endif // PSSHMEMINPUT_EXCEPTIONS_H
