//--------------------------------------------------------------------------
//
// Environment:
//      This software was developed for the BaBar collaboration.  If you
//      use all or part of it, please give an appropriate acknowledgement.
//
// Copyright Information:
//	Copyright (C) 2003	SLAC
//
//------------------------------------------------------------------------

#ifndef APPUTILS_APPCMDOPT_HH
#define APPUTILS_APPCMDOPT_HH

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppCmdOptBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdExceptions.h"
#include "AppUtils/AppCmdTypeTraits.h"

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
 *  @brief Command line option with a required argument of arbitrary type.
 *
 *  This class defines a command line option with an argument. This is a templated
 *  class parameterized by the type of the argument. Any type supported by the
 *  AppCmdTypeTraits can be used as a template parameter.
 *
 *  If option defines short (single-character) option name then it can be specified
 *  on the command line as '-oArgValue' or  '-o ArgValue' (two words). If option
 *  defines long (multi-character)  option name then it can be specified on the command
 *  line as '--option=ArgValue' or '--option ArgValue' or in options file as
 *  'option = ArgValue'.
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


template<class Type>
class AppCmdOpt : public AppCmdOptBase {

public:

  /**
   *  @brief Define an option with a required argument.
   *
   *  @deprecated This constructor is for backward-compatibility only, use constructor with
   *  optNames argument in the new code.
   *
   *  This constructor defines an option with both short name (-o) and long name
   *  (--option) which has a required argument. The argument is given to option as
   *  `-o value', `--option=value' on the command line or as `option = value' in
   *  the options file. After option is instantiated it has to be added to parser
   *  using AppCmdLine::addOption() method. To get current value of option argument
   *  use value() method.
   *
   *  @param[in] shortOpt    Short one-character option name
   *  @param[in] longOpt     Long option name (not including leading --)
   *  @param[in] name        Name for option argument, something like "path", "number", etc. Used
   *                         only for information purposes when usage() is called.
   *  @param[in] descr       Long description for the option, printed when usage() is called.
   *  @param[in] defValue    Value returned from value() if option is not specified on command line.
   */
  AppCmdOpt ( char shortOpt,
              const std::string& longOpt,
              const std::string& name,
              const std::string& descr,
              const Type& defValue ) ;

  /**
   *  @brief Define an option with a required argument.
   *
   *  This constructor can define option with both short name (-o) and long name (--option).
   *  All option names are defined via single constructor argument optNames which contains a
   *  comma-separated list of option names (like "option,o"). Single character becomes short
   *  name (-o), longer string becomes long name (--option).  The argument is given to option
   *  as `-o value', `--option=value' on the command line or as `option = value' in
   *  the options file. After option is instantiated it has to be added to parser
   *  using AppCmdLine::addOption() method. To get current value of option argument
   *  use value() method.
   *
   *  @param[in] optNames    Comma-separated option names.
   *  @param[in] name        Name for option argument, something like "path", "number", etc. Used
   *                         only for information purposes when usage() is called.
   *  @param[in] descr       Long description for the option, printed when usage() is called.
   *  @param[in] defValue    Value returned from value() if option is not specified on command line.
   */
  AppCmdOpt ( const std::string& optNames,
              const std::string& name,
              const std::string& descr,
              const Type& defValue ) ;

  /**
   *  @brief Define an option with a required argument.
   *
   *  @deprecated This constructor is for backward-compatibility only, use constructor with
   *  optNames argument in the new code.
   *
   *  This constructor defines an option with short name (-o) which has a required
   *  argument. The argument is given to option as `-o value' on the command line, option
   *  cannot be used in the options file. After option is instantiated it has to be added to
   *  parser using AppCmdLine::addOption() method. To get current value of option argument
   *  use value() method.
   *
   *  @param[in] shortOpt    Short one-character option name
   *  @param[in] name        Name for option argument, something like "path", "number", etc. Used
   *                         only for information purposes when usage() is called.
   *  @param[in] descr       Long description for the option, printed when usage() is called.
   *  @param[in] defValue    Value returned from value() if option is not specified on command line.
   */
  AppCmdOpt ( char shortOpt,
              const std::string& name,
              const std::string& descr,
              const Type& defValue ) ;

  /// Destructor
  virtual ~AppCmdOpt( ) {}

  /**
   *  True if the value of the option was changed from command line or from option file.
   */
  virtual bool valueChanged() const { return _changed ; }


  /**
   *  Return current value of the argument
   */
  virtual const Type& value() const { return _value ; }


  /**
   *  Return default value of the argument
   */
  const Type& defValue() const { return _defValue ; }

protected:

  // Helper functions

private:

  /**
   *  Returns true if option requires argument.
   */
  virtual bool hasArgument() const { return true; }

  /**
   *  @brief Set option's argument.
   *
   *  This method is called by parser when option is found on command line.
   *  The value string will be empty if hasArgument() is false.
   *  Shall throw an exception in case of value conversion error.
   *
   *  @throw AppCmdException Thrown if string to value conversion fails.
   */
  virtual void setValue( const std::string& value ) {
    _value = AppCmdTypeTraits<Type>::fromString ( value ) ;
    _changed = true ;
  }


  /**
   *  Reset option to its default value
   */
  virtual void reset() {
    _value = _defValue ;
    _changed = false ;
  }


  // Data members
  Type _value ;
  const Type _defValue ;
  bool _changed ;

  // This class is non-copyable
  AppCmdOpt( const AppCmdOpt& );
  AppCmdOpt& operator= ( const AppCmdOpt& );

};

// constructors
template <typename Type>
AppCmdOpt<Type>::AppCmdOpt ( char shortOpt,
                             const std::string& longOpt,
                             const std::string& name,
                             const std::string& descr,
                             const Type& defValue )
  : AppCmdOptBase(longOpt+","+std::string(1, shortOpt), name, descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

template <typename Type>
AppCmdOpt<Type>::AppCmdOpt ( const std::string& optNames,
                             const std::string& name,
                             const std::string& descr,
                             const Type& defValue )
  : AppCmdOptBase(optNames, name, descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

template <typename Type>
AppCmdOpt<Type>::AppCmdOpt ( char shortOpt,
                             const std::string& name,
                             const std::string& descr,
                             const Type& defValue )
  : AppCmdOptBase(std::string(1, shortOpt), name, descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

} // namespace AppUtils

#endif  // APPUTILS_APPCMDOPT_HH
