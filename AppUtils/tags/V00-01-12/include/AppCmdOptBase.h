#ifndef APPUTILS_APPCMDOPTBASE_HH
#define APPUTILS_APPCMDOPTBASE_HH

//--------------------------------------------------------------------------
//
// Environment:
//      This software was developed for the BaBar collaboration.  If you
//      use all or part of it, please give an appropriate acknowledgement.
//
// Copyright Information:
//      Copyright (C) 2003	SLAC
//
//------------------------------------------------------------------------

//-------------
// C Headers --
//-------------
extern "C" {
}

//---------------
// C++ Headers --
//---------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdExceptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace AppUtils {

/// @addtogroup AppUtils

/**
 *  @ingroup AppUtils
 *
 *  This is the base class for the optional arguments in the command line.
 *  Every optional argument must provide its short option name (for the -x
 *  form), long option name (for the --xxx form) and specify whether it takes
 *  any argument. options with argument should also give argument name. And
 *  every option provides short description.
 *
 *  Options with arguments can be secified on the command line as '-xArgValue',
 *  '-x ArgValue' (two words), '--xxxx=ArgValue' or '--xxxx ArgValue'.
 *
 *  Few options without argument could be combined into one word on the command
 *  line, e.g. -vvvxs.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2003		SLAC
 *
 *  @see AppCmdLine
 *  @see AppCmdOpt
 *  @see AppCmdOptIncr
 *  @see AppCmdOptToggle
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov	(originator)
 */

class AppCmdOptBase {

public:

  /// Destructor
  virtual ~AppCmdOptBase( ) throw() ;

  /**
   *  Returns true if option requires argument. Does not make sense for
   *  positional arguments.
   */
  virtual bool hasArgument() const throw() = 0 ;

  /**
   *  Get the name of the argument, only used if hasArgument() returns true
   */
  virtual const std::string& name() const throw() = 0 ;

  /**
   *  Get one-line description
   */
  virtual const std::string& description() const throw() = 0 ;

  /**
   *  Return short option symbol for -x option, or @c NULL if no short option
   */
  virtual char shortOption() const throw() = 0 ;

  /**
   *  Return long option symbol for --xxxxx option, or empty string
   */
  virtual const std::string& longOption() const throw() = 0 ;

  /**
   *  Set option's argument. The value string will be empty if hasArgument() is false
   *  Will throw an exception in case of conversion error
   */
  virtual void setValue( const std::string& value ) throw(AppCmdException) = 0 ;

  /**
   *  True if the value of the option was changed from command line.
   */
  virtual bool valueChanged () const throw() = 0 ;

  /**
   *  reset option to its default value
   */
  virtual void reset() throw() = 0 ;

protected:

  /**
   *  Constructor.
   */
  AppCmdOptBase() {}


private:

  // Friends

  // Note: if your class needs a copy constructor or an assignment operator,
  //  make one of the following public and implement it.
  AppCmdOptBase( const AppCmdOptBase& );                // Copy Constructor
  AppCmdOptBase& operator= ( const AppCmdOptBase& );    // Assignment op


};

} // namespace AppUtils

#endif // APPUTILS_APPCMDOPTBASE_HH
