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

  /// Destructor
  virtual ~AppCmdOptNamedValue( );

  /**
   *  Returns true if option requires argument. Does not make sense for
   *  positional arguments.
   */
  virtual bool hasArgument() const ;

  /**
   *  Get the name of the argument, only used if hasArgument() returns true
   */
  virtual const std::string& name() const ;

  /**
   *  Get one-line description
   */
  virtual const std::string& description() const ;

  /**
   *  Return short option symbol for -x option
   */
  virtual char shortOption() const ;

  /**
   *  Return long option symbol for --xxxxx option
   */
  virtual const std::string& longOption() const ;

  /**
   *  Set option's argument. The value string will be empty if hasArgument() is false
   */
  virtual bool setValue( const std::string& value ) ;

  /**
   *  True if the value of the option was changed from command line.
   */
  virtual bool valueChanged() const ;

  /**
   *  Return current value of the argument
   */
  virtual const Type& value() const ;

  /**
   *  Return default value of the argument
   */
  const Type& defValue() const { return _defValue ; }

  /**
   *  Reset option to its default value
   */
  virtual void reset() ;

  /// add string-value pair
  void add ( const std::string& key, const Type& value ) ;

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
