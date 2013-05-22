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

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace AppUtils {

/// @addtogroup AppUtils

/**
 *  @ingroup AppUtils
 *
 *  Option class for specifying size with a number+suffix. Acceptable
 *  suffixes are k, K, M, G.
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

  // option with both short and long names
  AppCmdOptSize ( char shortOpt,
                  const std::string& longOpt,
                  const std::string& name,
                  const std::string& descr,
                  value_type defValue ) ;
  // option with the long name only
  AppCmdOptSize ( const std::string& longOpt,
                  const std::string& name,
                  const std::string& descr,
                  value_type defValue ) ;
  // option with the short name only
  AppCmdOptSize ( char shortOpt,
                  const std::string& name,
                  const std::string& descr,
                  value_type defValue ) ;

  // Destructor
  virtual ~AppCmdOptSize () throw() ;

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
   *  Return current value of the option
   */
  virtual value_type value() const throw() ;

  /**
   *  Return default value of the argument
   */
  value_type defValue() const throw() { return _defValue ; }

  /**
   *  Reset option to its default value
   */
  virtual void reset() throw() ;

protected:

private:

  // Data members
  const char _shortOpt ;
  const std::string _longOpt ;
  const std::string _name ;
  const std::string _descr ;
  value_type _value ;
  const value_type _defValue ;
  bool _changed ;

  // Copy constructor and assignment are disabled by default
  AppCmdOptSize ( const AppCmdOptSize& ) ;
  AppCmdOptSize& operator = ( const AppCmdOptSize& ) ;

};

} // namespace AppUtils

#endif // APPUTILS_APPCMDOPTSIZE_H
