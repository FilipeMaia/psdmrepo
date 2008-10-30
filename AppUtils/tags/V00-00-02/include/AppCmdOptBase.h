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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------


/**
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
 *  @version $Id: AppCmdOptBase.hh,v 1.3 2004/08/24 17:04:23 salnikov Exp $
 *
 *  @author Andy Salnikov	(originator)
 */

namespace AppUtils {

class AppCmdOptBase {

public:

  /// Destructor
  virtual ~AppCmdOptBase( );

  /**
   *  Returns true if option requires argument. Does not make sense for
   *  positional arguments.
   */
  virtual bool hasArgument() const = 0 ;

  /**
   *  Get the name of the argument, only used if hasArgument() returns true
   */
  virtual const std::string& name() const = 0 ;

  /**
   *  Get one-line description
   */
  virtual const std::string& description() const = 0 ;

  /**
   *  Return short option symbol for -x option
   */
  virtual char shortOption() const = 0 ;

  /**
   *  Return long option symbol for --xxxxx option
   */
  virtual const std::string& longOption() const = 0 ;

  /**
   *  Set option's argument. The value string will be empty if hasArgument() is false
   */
  virtual bool setValue( const std::string& value ) = 0 ;

  /**
   *  True if the value of the option was changed from command line.
   */
  virtual bool valueChanged () const = 0 ;

  /**
   *  reset option to its default value
   */
  virtual void reset() = 0 ;

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
