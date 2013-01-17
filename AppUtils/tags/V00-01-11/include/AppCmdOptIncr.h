#ifndef APPUTILS_APPCMDOPTINCR_HH
#define APPUTILS_APPCMDOPTINCR_HH

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

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppCmdOptBase.h"

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
 *  This class represents a command line option without argument. Every
 *  appearance of the option on the command line will increment the current
 *  value of the option.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2003		SLAC
 *
 *  @see AppCmdOpt
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov	(originator)
 */

namespace AppUtils {

class AppCmdOptIncr : public AppCmdOptBase {

public:

  /**
   *  Make an option
   *
   *  @param shortOpt  short form of the option, single character
   *  @param longOpt   long form of the option, string without leading --
   *  @param descr     description, one-line string
   *  @param defValue  initial value of the option
   */
  AppCmdOptIncr ( char shortOpt,
                  const std::string& longOpt,
                  const std::string& descr,
                  int defValue = 0 ) ;
  // option with the long name only
  AppCmdOptIncr ( const std::string& longOpt,
                  const std::string& descr,
                  int defValue = 0 ) ;
  // option with the short name only
  AppCmdOptIncr ( char shortOpt,
                  const std::string& descr,
                  int defValue = 0 ) ;

  /// Destructor
  virtual ~AppCmdOptIncr( ) throw() ;

  /**
   *  Returns true if option requires argument. Does not make sense for
   *  positional arguments.
   */
  virtual bool hasArgument() const throw() ;

  /**
   *  Get the name of the argument, only used if hasArgument() returns true
   */
  virtual const std::string& name() const throw() ;

  /**
   *  Get one-line description
   */
  virtual const std::string& description() const throw() ;

  /**
   *  Return short option symbol for -x option, or '\0' if no short option
   */
  virtual char shortOption() const throw() ;

  /**
   *  Return long option symbol for --xxxxx option, or empty string
   */
  virtual const std::string& longOption() const throw() ;

  /**
   *  Set option's argument. The value string will be empty if hasArgument() is false
   */
  virtual void setValue( const std::string& value ) throw(AppCmdException) ;

  /**
   *  True if the value of the option was changed from command line.
   */
  virtual bool valueChanged() const throw() ;

  /**
   *  Return current value of the option
   */
  virtual int value() const throw() ;

  /**
   *  Return default value of the argument
   */
  int defValue() const throw() { return _defValue ; }

  /**
   *  Reset option to its default value
   */
  virtual void reset() throw() ;

protected:

  // Helper functions

private:

  // Friends

  // Data members
  const char _shortOpt ;
  const std::string _longOpt ;
  const std::string _name ;
  const std::string _descr ;
  int _value ;
  const int _defValue ;
  bool _changed ;

  // Note: if your class needs a copy constructor or an assignment operator,
  //  make one of the following public and implement it.
  AppCmdOptIncr( const AppCmdOptIncr& );                // Copy Constructor
  AppCmdOptIncr& operator= ( const AppCmdOptIncr& );    // Assignment op

};

} // namespace AppUtils

#endif // APPUTILS_APPCMDOPTINCR_HH
