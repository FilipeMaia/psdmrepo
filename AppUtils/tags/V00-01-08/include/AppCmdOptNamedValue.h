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

//-------------
// C Headers --
//-------------
extern "C" {
}

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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------


/**
 *  This class defines a command line option with argument. The argument is
 *  a string which is mapped to the value of the different type. Mapping is
 *  determined by user who should provide all accepted strings and their
 *  corresponding values via add() method.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2003		SLAC
 *
 *  @see AppCmdOptBase
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov	(originator)
 */

namespace AppUtils {

template<class Type>
class AppCmdOptNamedValue : public AppCmdOptBase {

public:

  /**
   *  Make an option with an argument
   */
  AppCmdOptNamedValue ( char shortOpt,
                        const std::string& longOpt,
                        const std::string& name,
                        const std::string& descr,
                        const Type& defValue ) ;
  // make option with long name only
  AppCmdOptNamedValue ( const std::string& longOpt,
                        const std::string& name,
                        const std::string& descr,
                        const Type& defValue ) ;
  // make option with short name only
  AppCmdOptNamedValue ( char shortOpt,
                        const std::string& name,
                        const std::string& descr,
                        const Type& defValue ) ;

  /// Destructor
  virtual ~AppCmdOptNamedValue( ) throw();

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
   *  Return short option symbol for -x option, or '\0' if no short option
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
  virtual const Type& value() const throw() ;

  /**
   *  Return default value of the argument
   */
  const Type& defValue() const throw() { return _defValue ; }

  /**
   *  Reset option to its default value
   */
  virtual void reset() throw() ;

  /// add string-value pair
  void add ( const std::string& key, const Type& value ) throw(std::exception) ;

protected:

  // Helper functions

private:

  // Types
  typedef std::map< std::string, Type > String2Value ;

  // Data members
  const char _shortOpt ;
  const std::string _longOpt ;
  const std::string _name ;
  const std::string _descr ;
  String2Value _str2value ;
  const Type _defValue ;
  Type _value ;
  bool _changed ;

  // Note: if your class needs a copy constructor or an assignment operator,
  //  make one of the following public and implement it.
  AppCmdOptNamedValue( const AppCmdOptNamedValue<Type>& );  // Copy Constructor
  AppCmdOptNamedValue<Type>& operator= ( const AppCmdOptNamedValue<Type>& );

};

} // namespace AppUtils

#include  "AppUtils/AppCmdOptNamedValue.icc"

#endif  // APPUTILS_APPCMDOPTNAMEDVALUE_HH
