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
 *  @brief Positional argument with the value of arbitrary type.
 *
 *  This class represents a single-word positional parameter in the command
 *  line. This is a templated class parameterized by the type of the parameter.
 *  Any type supported by the AppCmdTypeTraits is allowed as a template parameter.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgment.
 *
 *  Copyright (C) 2003          SLAC
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov       (originator)
 */

template<class Type>
class AppCmdArg : public AppCmdArgBase {

public:

  /**
   *  @brief Make a required positional argument.
   *
   *  The initial value will be default-constructed but it has to be specified on
   *  command line.
   *
   *  @param name  The name of the argument, like "file", "time", etc. This name
   *               is only used in the usage() to print info about this argument
   *  @param descr One-line description of the argument, used by usage()
   */
  AppCmdArg ( const std::string& name, const std::string& descr ) ;

  /**
   *  @brief Make an optional positional argument.
   *
   *  @param name  The name of the argument, like "file", "time", etc. This name
   *               is only used in the usage() to print info about this argument
   *  @param descr One-line description of the argument, used by usage()
   *  @param val   default value for this argument
   */
  AppCmdArg ( const std::string& name, const std::string& descr, const Type& val ) ;

  // Destructor
  virtual ~AppCmdArg( ) {}

  /**
   *  Return current value of the argument
   */
  virtual const Type& value() const { return _value ; }


  /**
   *  True if the value of the option was changed from command line. Only
   *  makes sense for "optional arguments", for required this should always
   *  return true.
   */
  virtual bool valueChanged() const { return _changed ; }


  /**
   *  Return default value of the argument
   */
  const Type& defValue() const { return _defValue ; }

protected:

  /**
   *  Is this argument required?
   */
  virtual bool isRequired() const { return _required ; }


  /**
   *  Get the name of the parameter
   */
  virtual const std::string& name() const { return _name ; }


  /**
   *  Get one-line description
   */
  virtual const std::string& description() const { return _descr ; }


  /**
   *  How many words from command line could this argument take? Single-word
   *  parameters should return 1. Parameters that can take the list of words
   *  Should return some big number. Note there is no function minWords() because
   *  it would always return 1.
   */
  virtual size_t maxWords () const { return 1 ; }

  /**
   *  Set the value of the argument. Throws an exception in case of
   *  type conversion errors.
   *
   *  @param begin  "pointer" to a starting word
   *  @param end    "pointer" behind the last word. For single-word parameters
   *                (++begin==end) will be true. For multi-word parameters the exact
   *                number of words given will depend on the number of words in the
   *                command and the number of positional arguments.
   *
   *  @return The number of consumed words. If it is negative then error has occurred.
   */
  virtual int setValue ( StringList::const_iterator begin,
                         StringList::const_iterator end ) ;

  /**
   *  Reset argument to its default value
   */
  virtual void reset() {
    _value = _defValue ;
    _changed = false ;
  }


private:

  // Friends

  // Data members
  const std::string _name ;
  const std::string _descr ;
  bool _required ;
  const Type _defValue ;
  Type _value ;
  bool _changed ;

  // This class is non-copyable
  AppCmdArg( const AppCmdArg& );
  AppCmdArg& operator= ( const AppCmdArg& );

};

/*
 *  Make a required positional argument
 */
template <typename Type>
AppCmdArg<Type>::AppCmdArg ( const std::string& name, const std::string& descr )
  : AppCmdArgBase()
  , _name(name)
  , _descr(descr)
  , _required(true)
  , _defValue()
  , _value()
  , _changed(false)
{
}

/*
 *  Make an optional positional argument
 */
template <typename Type>
AppCmdArg<Type>::AppCmdArg ( const std::string& name, const std::string& descr, const Type& val )
  : AppCmdArgBase()
  , _name(name)
  , _descr(descr)
  , _required(false)
  , _defValue(val)
  , _value(val)
  , _changed(false)
{
}

/*
 *  Set the value of the argument.
 *
 *  @return The number of consumed words. If it is negative then error has occurred.
 */
template <typename Type>
int
AppCmdArg<Type>::setValue ( StringList::const_iterator begin,
                            StringList::const_iterator end )
{
  // sequence must be non-empty
  assert ( begin != end ) ;

  _value = AppCmdTypeTraits<Type>::fromString ( *begin ) ;
  _changed = true ;
  // only one string could be supplied
  assert ( ++ begin == end ) ;
  return 1 ;
}

} // namespace AppUtils

#endif  // APPUTILS_APPCMDARG_HH
