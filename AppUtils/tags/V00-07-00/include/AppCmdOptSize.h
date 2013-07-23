#ifndef APPUTILS_APPCMDOPTSIZE_H
#define APPUTILS_APPCMDOPTSIZE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdOptSize.
//
//------------------------------------------------------------------------

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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace AppUtils {
class AppCmdOptGroup;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace AppUtils {

/// @addtogroup AppUtils

/**
 *  @ingroup AppUtils
 *
 *  @brief Class defining option with argument for specifying file sizes/offsets.
 *
 *  This class represents option with required argument, argument value format is
 *  number+suffix. Acceptable suffixes are k, K, M, G. This option can be useful
 *  when specifying file size for example (e.g. --size-limit 1G).
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class AppCmdOptSize : public AppCmdOptBase {
public:

  typedef unsigned long long value_type ;

  /**
   *  @brief Define an option with a required argument.
   *
   *  This constructor can define option with both short name (-o) and long name (--option).
   *  All option names are defined via single constructor argument optNames which contains a
   *  comma-separated list of option names (like "option,o"). Single character becomes short
   *  name (-o), longer string becomes long name (--option). After option is instantiated it
   *  has to be added to parser using AppCmdLine::addOption() method. To get current value of
   *  option argument use value() method.
   *
   *  @param[in] optNames    Comma-separated option names.
   *  @param[in] name        Name for option argument, something like "path", "number", etc. Used
   *                         only for information purposes when usage() is called.
   *  @param[in] descr       Long description for the option, printed when usage() is called.
   *  @param[in] defValue    Value returned from value() if option is not specified on command line.
   */
  AppCmdOptSize(const std::string& optNames, const std::string& name, const std::string& descr, value_type defValue);

  /**
   *  @brief Define an option with a required argument.
   *
   *  This constructor can define option with both short name (-o) and long name (--option).
   *  All option names are defined via single constructor argument optNames which contains a
   *  comma-separated list of option names (like "option,o"). Single character becomes short
   *  name (-o), longer string becomes long name (--option).
   *  This constructor automatically adds instantiated option to a parser.
   *  To get current value of option argument use value() method.
   *  This method may throw an exception if the option name conflicts with the previously
   *  added options.
   *
   *  @param[in] group       Option group (or parser instance) to which this option will be added.
   *  @param[in] optNames    Comma-separated option names.
   *  @param[in] name        Name for option argument, something like "path", "number", etc. Used
   *                         only for information purposes when usage() is called.
   *  @param[in] descr       Long description for the option, printed when usage() is called.
   *  @param[in] defValue    Value returned from value() if option is not specified on command line.
   */
  AppCmdOptSize(AppCmdOptGroup& group, const std::string& optNames, const std::string& name, const std::string& descr,
      value_type defValue);

  // Destructor
  virtual ~AppCmdOptSize () ;

  /**
   *  True if the value of the option was changed from command line or from option file.
   */
  virtual bool valueChanged() const ;

  /**
   *  Return current value of the option
   */
  virtual value_type value() const ;

  /**
   *  Return default value of the argument
   */
  value_type defValue() const { return _defValue ; }

protected:

private:

  /**
   *  Returns true if option requires argument. Does not make sense for
   *  positional arguments.
   */
  virtual bool hasArgument() const ;

  /**
   *  Get one-line description, should be brief but informational, may include default
   *  or initial value for the option.
   */
  virtual std::string description() const;

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
  value_type _value ;
  const value_type _defValue ;
  bool _changed ;

  // This class in non-copyable
  AppCmdOptSize ( const AppCmdOptSize& ) ;
  AppCmdOptSize& operator = ( const AppCmdOptSize& ) ;

};

} // namespace AppUtils

#endif // APPUTILS_APPCMDOPTSIZE_H
