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
namespace AppUtils {
class AppCmdLine;
}

//		---------------------
// 		-- Class Interface --
//		---------------------


namespace AppUtils {

/// @addtogroup AppUtils

/**
 *  @ingroup AppUtils
 *
 *  @brief Option class for boolean (toggle-type) flag.
 *
 *  This class represents a command line option without argument. The option
 *  has boolean value which will change (flip between true and false) its value
 *  for every appearance of the  option in the command line, so even number
 *  of appearances is identical to not using an option at all.
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

class AppCmdOptToggle : public AppCmdOptBase {

public:

  /**
   *  @brief Define toggle-type option without argument.
   *
   *  @deprecated This constructor is for backward-compatibility only, use constructor with
   *  optNames argument in the new code.
   *
   *  This constructor defines an option with both short name (-o) and long name (--option).
   *  After option is instantiated it has to be added to parser using AppCmdLine::addOption()
   *  method. To get current value of option argument use value() method.
   *
   *  @param[in] shortOpt    Short one-character option name
   *  @param[in] longOpt     Long option name (not including leading --)
   *  @param[in] descr     description, one-line string
   *  @param[in] defValue  initial value of the option
   */
  AppCmdOptToggle(char shortOpt, const std::string& longOpt, const std::string& descr, bool defValue = false);

  /**
   *  @brief Define toggle-type option without argument.
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
  AppCmdOptToggle(const std::string& optNames, const std::string& descr, bool defValue = false);

  /**
   *  @brief Define toggle-type option without argument.
   *
   *  This constructor can define option with both short name (-o) and long name (--option).
   *  All option names are defined via single constructor argument optNames which contains a
   *  comma-separated list of option names (like "option,o"). Single character becomes short
   *  name (-o), longer string becomes long name (--option).
   *  This constructor automatically adds instantiated option to a parser.
   *  This method may throw an exception if the option name conflicts with the previously
   *  added options.
   *
   *  @param[in] parser      Parser instance to which this option will be added.
   *  @param[in] optNames    Comma-separated option names.
   *  @param[in] descr     description, one-line string
   *  @param[in] defValue  initial value of the option
   */
  AppCmdOptToggle(AppCmdLine& parser, const std::string& optNames, const std::string& descr, bool defValue = false);

  /**
   *  @brief Define toggle-type option without argument.
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
  AppCmdOptToggle(char shortOpt, const std::string& descr, bool defValue = false);

  /// Destructor
  virtual ~AppCmdOptToggle( );

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

protected:

  // Helper functions

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

  // This class in non-copyable
  AppCmdOptToggle( const AppCmdOptToggle& );
  AppCmdOptToggle& operator= ( const AppCmdOptToggle& );

};

} // namespace AppUtils

#endif // APPUTILS_APPCMDOPTTOGGLE_HH
