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

#ifndef APPUTILS_APPCMDARG_HH
#define APPUTILS_APPCMDARG_HH

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
#include "AppUtils/AppCmdArgBase.h"

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
 *  This class represents a single-word positional parameter in the command
 *  line. This is a templated class parameterized by the type of the parameter.
 *  Any type supported by the AppCmdTypeTraits is allowed as a template parameter.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2003		SLAC
 *
 *  @see AppCmdArgBase
 *
 *  @version $Id: AppCmdArg.hh,v 1.3 2004/08/24 17:04:22 salnikov Exp $
 *
 *  @author Andy Salnikov	(originator)
 */

namespace AppUtils {

template<class Type>
class AppCmdArg : public AppCmdArgBase {

public:

  /**
   *  Make a required positional argument. The value will be default-constructed.
   *
   *  @param name  The name of the argument, like "file", "time", etc. This name
   *               is only used in the usage() to print info about this argument
   *  @param descr One-line description of the argument, used by usage()
   */
  AppCmdArg ( const std::string& name, const std::string& descr ) ;

  /**
   *  Make an optional positional argument.
   *
   *  @param name  The name of the argument, like "file", "time", etc. This name
   *               is only used in the usage() to print info about this argument
   *  @param descr One-line description of the argument, used by usage()
   *  @param val   default value for this argument
   */
  AppCmdArg ( const std::string& name, const std::string& descr, const Type& val ) ;

  // Destructor
  virtual ~AppCmdArg( );

  /**
   *  Is this argument required?
   */
  virtual bool isRequired() const ;

  /**
   *  Get the name of the paramater
   */
  virtual const std::string& name() const ;

  /**
   *  Get one-line description
   */
  virtual const std::string& description() const ;

  /**
   *  How many words from command line could this argument take? Single-word
   *  parameters should return 1. Parameters that can take the list of words
   *  Should return some big number. Note there is no function minWords() because
   *  it would always return 1.
   */
  virtual size_t maxWords () const ;

  /**
   *  Set the value of the argument.
   *
   *  @param begin  "pointer" to a starting word
   *  @param end    "pointer" behind the last word. For single-word parameters
   *                (++begin==end) will be true. For multi-word parameters the exact
   *                number of words given will depend on the number of words in the
   *                command and the number of positional arguments.
   *
   *  @return The number of consumed words. If it is negative then error has occured.
   */
  virtual int setValue ( StringList::const_iterator begin,
                         StringList::const_iterator end ) ;

  /**
   *  True if the value of the option was changed from command line. Only
   *  makes sense for "optionsl arguments", for required this will always
   *  return true.
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
   *  Reset argument to its default value
   */
  virtual void reset() ;

protected:

  // Helper functions

private:

  // Friends

  // Data members
  const std::string _name ;
  const std::string _descr ;
  bool _required ;
  const Type _defValue ;
  Type _value ;
  bool _changed ;

  // Note: if your class needs a copy constructor or an assignment operator,
  //  make one of the following public and implement it.
  AppCmdArg( const AppCmdArg<Type>& );  // Copy Constructor
  AppCmdArg<Type>& operator= ( const AppCmdArg<Type>& );

};

} // namespace AppUtils

#include  "AppUtils/AppCmdArg.icc"

#endif  // APPUTILS_APPCMDARG_HH
