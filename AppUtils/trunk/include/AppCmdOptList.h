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

#ifndef APPUTILS_APPCMDOPTLIST_HH
#define APPUTILS_APPCMDOPTLIST_HH

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
 *  @brief Option class collecting arguments into list of values.
 *
 *  This class defines a command line option with argument. This is a templated
 *  class parameterized by the type of the argument. Any type supported by the
 *  AppCmdTypeTraits can be used as a template parameter.
 *
 *  Options of this type are used to collect multiple option arguments into
 *  a list of values. The argument string associated with the option is split
 *  into multiple pieces on separator character defined in constructor (default
 *  is comma). Each piece is converted to final type (template type) and appended
 *  to a sequence which becomes a value of the option. If option appears multiple
 *  times on the command line, all option values will be collected in one
 *  sequences.
 *
 *  If for example Option of type @c AppCmdOptList<int> is defined with name -n then
 *  all three following command lines will produce identical option value:
 *    @li -n 1 -n 2 -n 5 -n 100
 *    @li -n 1,2,5,100
 *    @li -n 1,2 -n 5,100
 *
 *  and the resulting value of the option will be a list with values [1, 2, 5, 100].
 *
 *  Initial value of the option is always an empty list.
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
class AppCmdOptList : public AppCmdOptBase {

public:

  typedef std::list<Type> container ;
  typedef typename container::const_iterator const_iterator ;
  typedef typename container::size_type size_type ;

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
   *  @param[in] separator   Separator character for splitting argument into sequences of values.
   */
  AppCmdOptList(char shortOpt, const std::string& longOpt, const std::string& name, const std::string& descr,
      char separator = ',');

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
   *  @param[in] separator   Separator character for splitting argument into sequences of values.
   */
  AppCmdOptList(const std::string& optNames, const std::string& name, const std::string& descr,
      char separator = ',');

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
   *  @param[in] separator   Separator character for splitting argument into sequences of values.
   */
  AppCmdOptList(AppCmdLine& parser, const std::string& optNames, const std::string& name, const std::string& descr,
      char separator = ',');

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
   *  @param[in] separator   Separator character for splitting argument into sequences of values.
   */
    AppCmdOptList(char shortOpt, const std::string& name, const std::string& descr, char separator = ',');

  /// Destructor
  virtual ~AppCmdOptList( ) {}

  /**
   *  True if the value of the option was changed from command line.
   */
  virtual bool valueChanged() const { return _changed ; }

  /**
   *  Return current value of the argument
   */
  virtual const container& value() const { return _value ; }

  /**
   *  Return iterator to the begin/end of sequence
   */
  virtual const_iterator begin() const { return _value.begin() ; }
  virtual const_iterator end() const { return _value.end() ; }

  /**
   *  Other usual container stuff
   */
  size_type size() const { return _value.size() ; }
  bool empty() const { return _value.empty() ; }

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
   *  Reset option to its default value, clear changed flag
   */
  virtual void reset() {
    _value.clear() ;
    _changed = false ;
  }

  // Data members
  const char _separator ;
  container _value ;
  bool _changed ;

  // This class in non-copyable
  AppCmdOptList( const AppCmdOptList& );
  AppCmdOptList& operator= ( const AppCmdOptList& );

};

template <typename Type>
AppCmdOptList<Type>::AppCmdOptList(char shortOpt, const std::string& longOpt, const std::string& name,
    const std::string& descr, char separator)
  : AppCmdOptBase(longOpt+","+std::string(1, shortOpt), name, descr)
  , _separator(separator)
  , _value()
  , _changed(false)
{
}

template <typename Type>
AppCmdOptList<Type>::AppCmdOptList(const std::string& optNames, const std::string& name, const std::string& descr,
    char separator)
  : AppCmdOptBase(optNames, name, descr)
  , _separator(separator)
  , _value()
  , _changed(false)
{
}

template <typename Type>
AppCmdOptList<Type>::AppCmdOptList(AppCmdLine& parser, const std::string& optNames, const std::string& name,
    const std::string& descr, char separator)
  : AppCmdOptBase(optNames, name, descr)
  , _separator(separator)
  , _value()
  , _changed(false)
{
  parser.addOption(*this);
}

template <typename Type>
AppCmdOptList<Type>::AppCmdOptList(char shortOpt, const std::string& name, const std::string& descr, char separator)
  : AppCmdOptBase(std::string(1, shortOpt), name, descr)
  , _separator(separator)
  , _value()
  , _changed(false)
{
}

/**
 *  Set the value of the argument.
 */
template <typename Type>
void
AppCmdOptList<Type>::setValue ( const std::string& value )
{
  container localCont ;

  std::string::const_iterator pos = value.begin() ;
  do {

    // get next item from the string
    std::string::const_iterator pos1 = std::find ( pos, value.end(), _separator ) ;
    std::string item ( pos, pos1 ) ;
    // this may thrpw
    Type res = AppCmdTypeTraits<Type>::fromString ( item ) ;
    localCont.push_back( res ) ;

    // advance
    pos = pos1 ;
    if ( pos != value.end() ) ++ pos ;

  } while ( pos != value.end() ) ;

  // copy from local container to value
  std::copy ( localCont.begin(), localCont.end(), std::back_inserter(_value) ) ;
  _changed = true ;
}

} // namespace AppUtils

#endif  // APPUTILS_APPCMDOPTLIST_HH
