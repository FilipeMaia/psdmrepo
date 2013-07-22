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
 *  @brief Class for integer increment (counter) option.
 *
 *  This class represents a command line option without argument. Every
 *  appearance of the option on the command line will increment the current
 *  value of the option. Initial value of the option can be set in constructor.
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

class AppCmdOptIncr : public AppCmdOptBase {

public:

  /**
   *  @brief Define incremental option without argument.
   *
   *  This constructor can define option with both short name (-o) and long name (--option).
   *  All option names are defined via single constructor argument optNames which contains a
   *  comma-separated list of option names (like "option,o"). Single character becomes short
   *  name (-o), longer string becomes long name (--option).
   *  After option is instantiated it has to be added to parser using AppCmdLine::addOption()
   *  method. To get current value of option argument use value() method.
   *
   *  @param[in] optNames    Comma-separated option names.
   *  @param[in] descr       Long description for the option, printed when usage() is called.
   *  @param[in] defValue    Initial value of the option
   */
  AppCmdOptIncr(const std::string& optNames, const std::string& descr, int defValue = 0);

  /**
   *  @brief Define incremental option without argument.
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
   *  @param[in] descr       Long description for the option, printed when usage() is called.
   *  @param[in] defValue    Initial value of the option
   *
   *  @throw AppCmdException or a subclass of it.
   */
  AppCmdOptIncr(AppCmdOptGroup& group, const std::string& optNames, const std::string& descr, int defValue = 0);

  /// Destructor
  virtual ~AppCmdOptIncr( ) ;

  /**
   *  True if the value of the option was changed from command line or from option file.
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

protected:

  // Helper functions

private:

  /**
   *  Returns true if option requires argument. Does not make sense for
   *  positional arguments.
   */
  virtual bool hasArgument() const ;

  /**
   *  Set option's argument. The value string will be empty if hasArgument() is false
   */
  virtual void setValue( const std::string& value ) ;

  /**
   *  Reset option to its default value
   */
  virtual void reset() ;


  // Data members
  int _value ;
  const int _defValue ;
  bool _changed ;

  // This class in non-copyable
  AppCmdOptIncr(const AppCmdOptIncr&);
  AppCmdOptIncr& operator=(const AppCmdOptIncr&);

};

} // namespace AppUtils

#endif // APPUTILS_APPCMDOPTINCR_HH
