#ifndef CONFIGSVC_EXCEPTIONS_H
#define CONFIGSVC_EXCEPTIONS_H

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
#include <stdexcept>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace ConfigSvc {

/// @addtogroup ConfigSvc

/**
 *  @ingroup ConfigSvc
 *
 *  Exception classes for ConfigSvc package.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class Exception : public std::runtime_error {
public:

  /// Constructor takes the reason for an exception
  Exception ( const std::string& what ) ;

};

// exception thrown when trying to use service before it was initialized
class ExceptionNotInitialized : public Exception {
public:

  ExceptionNotInitialized () ;

};

// exception thrown when trying to re-initialize service
class ExceptionInitialized : public Exception {
public:

  ExceptionInitialized () ;

};

// exception thrown when config file is not there or unreadable
class ExceptionFileMissing : public Exception {
public:

  /// Constructor takes the name of the file
  ExceptionFileMissing ( const std::string& file ) ;

};

// exception thrown when config file is not there or unreadable
class ExceptionFileRead : public Exception {
public:

  /// Constructor takes the name of the file
  ExceptionFileRead ( const std::string& file ) ;

};

// exception thrown when config file is not there or unreadable
class ExceptionSyntax : public Exception {
public:

  /// Constructor takes the name of the file and line number
  ExceptionSyntax ( const std::string& file, int lineno, const std::string& what ) ;

};

// thrown when specified section or parameter name is not found
class ExceptionMissing : public Exception {
public:

  /// Constructor takes the name of the section and parameter
  ExceptionMissing ( const std::string& section, 
      const std::string& parameter ) ;

};

// thrown when conversion from string to numbers has failed
class ExceptionCvtFail : public Exception {
public:

  /// Constructor takes the name of the section, parameter and string value
  ExceptionCvtFail (const std::string& section, const std::string& parameter,
      const std::string& string);

};

} // namespace ConfigSvc

#endif // CONFIGSVC_EXCEPTIONS_H
