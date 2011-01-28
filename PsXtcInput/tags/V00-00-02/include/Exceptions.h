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

namespace PsXtcInput {

class Exception : public std::runtime_error {
public:

  // Constructor
  Exception ( const std::string& className, const std::string& what ) ;

};

// thrown for empty file list
class EmptyFileList : public Exception {
public:

  EmptyFileList()
    : Exception( "EmptyFileList", "No input file names specified" ) {}

};

// thrown for empty file
class EmptyInput : public Exception {
public:

  EmptyInput()
    : Exception( "EmptyInput", "XTC file(s) is empty" ) {}

};

} // namespace PsXtcInput

#endif // PSXTCINPUT_EXCEPTIONS_H
