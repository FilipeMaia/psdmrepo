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

#ifndef APPUTILS_APPCMDOPTNAMEDVALUE_HH
#define APPUTILS_APPCMDOPTNAMEDVALUE_HH

//---------------
// C++ Headers --
//---------------
#include <map>

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppCmdOptBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdLine.h"
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
 *  @brief Class for option type which maps strings to values.
 *
 *  This class defines a command line option with argument. The argument is
 *  a string which is mapped to the value of the option template type. Mapping
 *  is determined by user who should provide all accepted strings and their
 *  corresponding values via add() method.
 *
 *  Typical use case for this class could be:
 *
 *  @code
 *  AppCmdOptNamedValue<Color::Enum> colorOpt("color,c", "COLOR", "Specifies color name, def: black", Color::Black);
 *  colorOpt.add("black", Color::Black);
 *  colorOpt.add("red", Color::Red);
 *  colorOpt.add("green", Color::Green);
 *  colorOpt.add("blue", Color::Blue);
 *  @endcode
 *
 *  Template argument could potentially be any type, typically it is enum, but any other
 *  Assignable and CopyConstructible type is acceptable.
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
class AppCmdOptNamedValue : public AppCmdOptBase {

public:

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
  AppCmdOptNamedValue(const std::string& optNames, const std::string& name, const std::string& descr,
      const Type& defValue);

  /**
   *  @brief Define an option with a required argument.
   *
   *  This constructor can define option with both short name (-o) and long name (--option).
   *  All option names are defined via single constructor argument optNames which contains a
   *  comma-separated list of option names (like "option,o"). Single character becomes short
   *  name (-o), longer string becomes long name (--option).  The argument is given to option
   *  as `-o value', `--option=value' on the command line or as `option = value' in
   *  the options file. This constructor automatically adds instantiated option to a parser.
   *  To get current value of option argument use value() method.
   *  This method may throw an exception if the option name conflicts with the previously
   *  added options.
   *
   *  @param[in] parser      Parser instance to which this option will be added.
   *  @param[in] optNames    Comma-separated option names.
   *  @param[in] name        Name for option argument, something like "path", "number", etc. Used
   *                         only for information purposes when usage() is called.
   *  @param[in] descr       Long description for the option, printed when usage() is called.
   *  @param[in] defValue    Value returned from value() if option is not specified on command line.
   */
  AppCmdOptNamedValue(AppCmdLine& parser, const std::string& optNames, const std::string& name,
      const std::string& descr, const Type& defValue);

  /// Destructor
  virtual ~AppCmdOptNamedValue( ) {}

  /**
   *  True if the value of the option was changed from command line.
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

  /// add string-value pair
  void add ( const std::string& key, const Type& value ) {
    typename String2Value::value_type thePair(key, value);
    _str2value.insert ( thePair );
  }


protected:

  // Helper functions

private:

  /**
   *  Returns true if option requires argument. Does not make sense for
   *  positional arguments.
   */
  virtual bool hasArgument() const { return true ; }

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
  virtual void reset() {
    _value = _defValue ;
    _changed = false ;
  }


  // Types
  typedef std::map< std::string, Type > String2Value ;

  // Data members
  String2Value _str2value ;
  const Type _defValue ;
  Type _value ;
  bool _changed ;

  // This class in non-copyable
  AppCmdOptNamedValue( const AppCmdOptNamedValue& );
  AppCmdOptNamedValue& operator= ( const AppCmdOptNamedValue& );

};

template <typename Type>
AppCmdOptNamedValue<Type>::AppCmdOptNamedValue(const std::string& optNames, const std::string& name,
    const std::string& descr, const Type& defValue)
  : AppCmdOptBase(optNames, name, descr)
  , _str2value()
  , _defValue(defValue)
  , _value(defValue)
  , _changed(false)
{
}

template <typename Type>
AppCmdOptNamedValue<Type>::AppCmdOptNamedValue(AppCmdLine& parser, const std::string& optNames, const std::string& name,
    const std::string& descr, const Type& defValue)
  : AppCmdOptBase(optNames, name, descr)
  , _str2value()
  , _defValue(defValue)
  , _value(defValue)
  , _changed(false)
{
  parser.addOption(*this);
}

/*
 *  Set the value of the argument.
 *
 *  @return The number of consumed words. If it is negative then error has occured.
 */
template <typename Type>
void
AppCmdOptNamedValue<Type>::setValue ( const std::string& valueStr )
{
  typename String2Value::const_iterator it = _str2value.find ( valueStr ) ;
  if ( it == _str2value.end() ) throw AppCmdTypeCvtException ( valueStr, "<map type>" ) ;

  _value = it->second ;
  _changed = true ;
}

} // namespace AppUtils

#endif  // APPUTILS_APPCMDOPTNAMEDVALUE_HH
