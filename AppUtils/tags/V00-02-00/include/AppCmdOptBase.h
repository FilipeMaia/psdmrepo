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

namespace AppUtils {

/// @addtogroup AppUtils

/**
 *  @ingroup AppUtils
 *
 *  @brief Base class for all command line option types.
 *
 *  This is the base class for the options in the command line.
 *  Options may provide short option name (for the -x form), or long option name
 *  (for the --xxx form), or both and specify whether it takes any argument.
 *  Options with argument should also give argument name, every option provides
 *  short description.
 *
 *  Options with arguments can be specified on the command line as '-xArgValue',
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
 *  @version $Id$
 *
 *  @author Andy Salnikov	(originator)
 */

class AppCmdOptBase {

public:

  /// Destructor
  virtual ~AppCmdOptBase( ) ;

  /**
   *  True if the value of the option was changed from command line.
   */
  virtual bool valueChanged () const = 0 ;

protected:

  // The methods below are protected as they do need to be exposed
  // to user, this interface should only be used by AppCmdLine which
  // is declared as friend. Subclasses may use these methods as well
  // for implementing their own functionality or override them.

  /**
   *  Returns true if option requires argument.
   */
  virtual bool hasArgument() const = 0 ;

  /**
   *  @brief Get the name of the argument.
   *
   *  Typically specifies the meaning of the argument value like "PATH", "COUNT", etc.,
   *  but may also give some special semantics, for example "(incr)" for increment
   *  options.
   */
  virtual const std::string& name() const { return _name; }

  /**
   *  Get one-line description, should be brief but informational.
   */
  virtual const std::string& description() const { return _descr; }

  /**
   *  Return short option symbol for -x option, or @c NUL if no short option is defined.
   */
  virtual char shortOption() const { return _shortOpt; }

  /**
   *  Return long option symbol for --xxxxx option format, or empty string in no long option is defined.
   */
  virtual const std::string& longOption() const { return _longOpt; }

  /**
   *  @brief Set option's argument.
   *
   *  This method is called by parser when option is found on command line.
   *  The value string will be empty if hasArgument() is false.
   *  Shall throw an exception in case of value conversion error.
   *
   *  @throw AppCmdException Thrown if string to value conversion fails.
   */
  virtual void setValue( const std::string& value ) = 0 ;

  /**
   *  reset option to its default value
   */
  virtual void reset() = 0 ;

  /**
   *  @brief Define an option.
   *
   *  This constructor can define option with both short name (-o) and long name (--option).
   *  All option names are defined via single constructor argument optNames which contains a
   *  comma-separated list of option names (like "option,o"). Single character becomes short
   *  name (-x), longer string becomes long name (--option).
   *
   *  @param[in] optNames    Comma-separated option
   *  @param[in] name        Name for option argument, something like "path", "number", etc. Used
   *                         only for information purposes when usage() is called.
   *  @param[in] descr       Long description for the option, printed when usage() is called.
   */
  AppCmdOptBase(const std::string& optNames,
      const std::string& name,
      const std::string& descr);

private:

  // All private methods are accessible to the parser
  friend class AppCmdLine;

  char _shortOpt ;
  std::string _longOpt ;
  const std::string _name ;
  const std::string _descr ;

  // This class is non-copyable
  AppCmdOptBase( const AppCmdOptBase& );
  AppCmdOptBase& operator= ( const AppCmdOptBase& );

};

} // namespace AppUtils

#endif // APPUTILS_APPCMDOPTBASE_HH
