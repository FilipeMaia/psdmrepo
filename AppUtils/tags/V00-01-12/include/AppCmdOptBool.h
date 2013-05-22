#ifndef APPUTILS_APPCMDOPTBOOL_HH
#define APPUTILS_APPCMDOPTBOOL_HH

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

namespace AppUtils {

/// @addtogroup AppUtils

/**
 *  @ingroup AppUtils
 *
 *  This class represents a command line option without argument. The option
 *  has boolean value which will change its value for the first appearance of the
 *  option in the command line.
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

class AppCmdOptBool : public AppCmdOptBase {

public:

  /**
   *  Make a toggle option.
   *
   *  @param shortOpt  short form of the option, single character
   *  @param longOpt   long form of the option, string without leading --
   *  @param descr     description, one-line string
   *  @param defValue  initial value of the option
   */
  AppCmdOptBool ( char shortOpt,
                  const std::string& longOpt,
                  const std::string& descr,
                  bool defValue = false ) ;
  // make option with long name only
  AppCmdOptBool ( const std::string& longOpt,
                  const std::string& descr,
                  bool defValue = false ) ;
  // make option with short name only
  AppCmdOptBool ( char shortOpt,
                  const std::string& descr,
                  bool defValue = false ) ;

  /// Destructor
  virtual ~AppCmdOptBool( ) throw();

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
   *  Return short option symbol for -x option, or @c NULL if no short option
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
   *  Return current value of the argument
   */
  virtual bool value() const throw() ;

  /**
   *  Return default value of the argument
   */
  bool defValue() const throw() { return _defValue ; }

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
  bool _value ;
  const bool _defValue ;
  bool _changed ;

  // Note: if your class needs a copy constructor or an assignment operator,
  //  make one of the following public and implement it.
  AppCmdOptBool( const AppCmdOptBool& );                // Copy Constructor
  AppCmdOptBool& operator= ( const AppCmdOptBool& );    // Assignment op

};

} // namespace AppUtils

#endif // APPUTILS_APPCMDOPTBOOL_HH
