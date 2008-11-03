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
 *  @version $Id: AppCmdOptIncr.hh,v 1.3 2004/08/24 17:04:23 salnikov Exp $
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

  /// Destructor
  virtual ~AppCmdOptIncr( );

  /**
   *  Returns true if option requires argument. Does not make sense for
   *  positional arguments.
   */
  virtual bool hasArgument() const ;

  /**
   *  Get the name of the argument, only used if hasArgument() returns true
   */
  virtual const std::string& name() const ;

  /**
   *  Get one-line description
   */
  virtual const std::string& description() const ;

  /**
   *  Return short option symbol for -x option
   */
  virtual char shortOption() const ;

  /**
   *  Return long option symbol for --xxxxx option
   */
  virtual const std::string& longOption() const ;

  /**
   *  Set option's argument. The value string will be empty if hasArgument() is false
   */
  virtual bool setValue( const std::string& value ) ;

  /**
   *  True if the value of the option was changed from command line.
   */
  virtual bool valueChanged() const ;

  /**
   *  Return current value of the option
   */
  virtual int value() const ;

  /**
   *  Return default value of the argument
   */
  int defValue() const { return _defValue ; }

  /**
   *  Reset option to its default value
   */
  virtual void reset() ;

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
