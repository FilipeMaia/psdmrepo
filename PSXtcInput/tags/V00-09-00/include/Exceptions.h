#ifndef PSXTCINPUT_EXCEPTIONS_H
#define PSXTCINPUT_EXCEPTIONS_H

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
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace PSXtcInput {

class Exception : public ErrSvc::Issue {
public:

  // Constructor
  Exception ( const ErrSvc::Context& ctx, 
              const std::string& className, 
              const std::string& what ) ;

};

// thrown for empty file list
class EmptyFileList : public Exception {
public:

  EmptyFileList(const ErrSvc::Context& ctx)
    : Exception( ctx, "EmptyFileList", "No input file names specified" ) {}

};

// thrown for empty file
class EmptyInput : public Exception {
public:

  EmptyInput(const ErrSvc::Context& ctx)
    : Exception( ctx, "EmptyInput", "XTC file(s) is empty" ) {}

};

} // namespace PSXtcInput

#endif // PSXTCINPUT_EXCEPTIONS_H
