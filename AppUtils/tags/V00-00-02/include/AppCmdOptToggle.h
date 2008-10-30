#ifndef APPUTILS_APPCMDOPTTOGGLE_HH
#define APPUTILS_APPCMDOPTTOGGLE_HH

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
 *  This class represents a command line option without argument. The option
 *  has boolean value which will change its value for every appearance of the
 *  option in the command line.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2003		SLAC
 *
 *  @see AppCmdOpt
 *
 *  @version $Id: AppCmdOptToggle.hh,v 1.3 2004/08/24 17:04:23 salnikov Exp $
 *
 *  @author Andy Salnikov	(originator)
 */

namespace AppUtils {

class AppCmdOptToggle : public AppCmdOptBase {

public:

  /**
   *  Make a toggle option.
   *
   *  @param shortOpt  short form of the option, single character
   *  @param longOpt   long form of the option, string without leading --
   *  @param descr     description, one-line string
   *  @param defValue  initial value of the option
   */
  AppCmdOptToggle ( char shortOpt,
		    const std::string& longOpt,
		    const std::string& descr,
		    bool defValue = false ) ;

  /// Destructor
  virtual ~AppCmdOptToggle( );

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
   *  Return current value of the argument
   */
  virtual bool value() const ;

  /**
   *  Return default value of the argument
   */
  bool defValue() const { return _defValue ; }

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
  bool _value ;
  const bool _defValue ;
  bool _changed ;

  // Note: if your class needs a copy constructor or an assignment operator,
  //  make one of the following public and implement it.
  AppCmdOptToggle( const AppCmdOptToggle& );                // Copy Constructor
  AppCmdOptToggle& operator= ( const AppCmdOptToggle& );    // Assignment op

};

} // namespace AppUtils

#endif // APPUTILS_APPCMDOPTTOGGLE_HH
