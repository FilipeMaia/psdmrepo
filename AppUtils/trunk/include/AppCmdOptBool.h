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
 *  @brief Option class for boolean flag.
 *
 *  This class represents a command line option without argument. The option
 *  has boolean value which will change its value for the first appearance of the
 *  option in the command line, more than one appearance of the option on
 *  command line is possible but is identical to just one appearance.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2003		SLAC
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov	(originator)
 */

class AppCmdOptBool : public AppCmdOptBase {

public:

  /**
   *  @brief Define boolean option without argument.
   *
   *  @deprecated This constructor is for backward-compatibility only, use constructor with
   *  optNames argument in the new code.
   *
   *  This constructor defines an option with both short name (-o) and long name (--option).
   *  After option is instantiated it has to be added to parser using AppCmdLine::addOption()
   *  method. To get current value of option argument use value() method.
   *
   *
   *  @param[in] shortOpt    Short one-character option name
   *  @param[in] longOpt     Long option name (not including leading --)
   *  @param[in] descr     description, one-line string
   *  @param[in] defValue  initial value of the option
   */
  AppCmdOptBool ( char shortOpt,
                  const std::string& longOpt,
                  const std::string& descr,
                  bool defValue = false ) ;

  /**
   *  @brief Define boolean option without argument.
   *
   *  This constructor can define option with both short name (-o) and long name (--option).
   *  All option names are defined via single constructor argument optNames which contains a
   *  comma-separated list of option names (like "option,o"). Single character becomes short
   *  name (-o), longer string becomes long name (--option).
   *  After option is instantiated it has to be added to parser using AppCmdLine::addOption()
   *  method. To get current value of option argument use value() method.
   *
   *  @param[in] optNames    Comma-separated option names.
   *  @param[in] descr     description, one-line string
   *  @param[in] defValue  initial value of the option
   */
  AppCmdOptBool ( const std::string& optNames,
                  const std::string& descr,
                  bool defValue = false ) ;

  /**
   *  @brief Define boolean option without argument.
   *
   *  @deprecated This constructor is for backward-compatibility only, use constructor with
   *  optNames argument in the new code.
   *
   *  This constructor defines an option with short name (-o) only.
   *  After option is instantiated it has to be added to parser using AppCmdLine::addOption()
   *  method. To get current value of option argument use value() method.
   *
   *  @param[in] shortOpt  short form of the option, single character
   *  @param[in] descr     description, one-line string
   *  @param[in] defValue  initial value of the option
   */
  AppCmdOptBool ( char shortOpt,
                  const std::string& descr,
                  bool defValue = false ) ;

  /// Destructor
  virtual ~AppCmdOptBool( );

  /**
   *  True if the value of the option was changed from command line or from option file.
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

protected:

private:

  /**
   *  Returns true if option requires argument. Does not make sense for
   *  positional arguments.
   */
  virtual bool hasArgument() const ;

  /**
   *  @brief Set option's argument.
   *
   *  This method is called by parser when option is found on command line.
   *  The value string will be empty if hasArgument() is false.
   *  Shall throw an exception in case of value conversion error.
   *
   *  @throw AppCmdException Thrown if string to value conversion fails.
   */
  virtual void setValue( const std::string& value ) ;

  /**
   *  Reset option to its default value
   */
  virtual void reset() ;


  // Data members
  bool _value ;
  const bool _defValue ;
  bool _changed ;

  // This class is non-copyable
  AppCmdOptBool( const AppCmdOptBool& );
  AppCmdOptBool& operator= ( const AppCmdOptBool& );

};

} // namespace AppUtils

#endif // APPUTILS_APPCMDOPTBOOL_HH
