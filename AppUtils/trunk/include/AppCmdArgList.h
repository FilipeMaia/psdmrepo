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

#ifndef APPUTILS_APPCMDARGLIST_HH
#define APPUTILS_APPCMDARGLIST_HH

//---------------
// C++ Headers --
//---------------
#include <vector>
#include <cassert>

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppCmdArgBase.h"

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
 *  @brief Positional argument class collecting arguments into list of values.
 *
 *  This class represents a multi-word positional parameter in the command
 *  line. This is a templated class parameterized by the type of the parameter.
 *  Any type supported by the AppCmdTypeTraits is allowed as a template parameter.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2003		SLAC
 *
 *  @see AppCmdArgListBase
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov	(originator)
 */

template<class Type>
class AppCmdArgList : public AppCmdArgBase {

public:

  typedef std::vector<Type> container ;
  typedef typename container::const_iterator const_iterator ;
  typedef typename container::size_type size_type ;

  /**
   *  @brief Make a required positional argument.
   *
   *  Required argument will need at least one word on the command line corresponding to it.
   *  After argument is instantiated it has to be added to parser using
   *  AppCmdLine::addArgument() method.
   *
   *  @param name  The name of the argument, like "files", "time", etc. This name
   *               is only used in the usage() to print info about this argument
   *  @param descr One-line description of the argument, used by usage()
   */
  AppCmdArgList(const std::string& name, const std::string& descr);

  /**
   *  @brief Make an optional positional argument.
   *
   *  Optional argument could consume 0 or more words from command line. If there are no words
   *  on the command line for this argument then its value will be a default value provided
   *  as argument to this constructor. If there are words on command line corresponding
   *  to this argument then their values replace default value, but do not extend it.
   *  After argument is instantiated it has to be added to parser using
   *  AppCmdLine::addArgument() method.
   *
   *  @param name  The name of the argument, like "files", "time", etc. This name
   *               is only used in the usage() to print info about this argument
   *  @param descr One-line description of the argument, used by usage()
   *  @param val   Default value for this argument
   */
  AppCmdArgList(const std::string& name, const std::string& descr, const container& val);

  /**
   *  @brief Make a required positional argument.
   *
   *  Required argument will need at least one word on the command line corresponding to it.
   *  This constructor automatically add instantiated argument to parser.
   *  This method may throw an exception if, for example, you try to add required
   *  argument after optional one.
   *
   *  @param[in] parser Parser instance to which this argument will be added.
   *  @param name  The name of the argument, like "files", "time", etc. This name
   *               is only used in the usage() to print info about this argument
   *  @param descr One-line description of the argument, used by usage()
   */
  AppCmdArgList(AppCmdLine& parser, const std::string& name, const std::string& descr);

  /**
   *  @brief Make an optional positional argument.
   *
   *  Optional argument could consume 0 or more words from command line. If there are no words
   *  on the command line for this argument then its value will be a default value provided
   *  as argument to this constructor. If there are words on command line corresponding
   *  to this argument then their values replace default value, but do not extend it.
   *  This constructor automatically add instantiated argument to parser.
   *
   *  @param[in] parser Parser instance to which this argument will be added.
   *  @param name  The name of the argument, like "files", "time", etc. This name
   *               is only used in the usage() to print info about this argument
   *  @param descr One-line description of the argument, used by usage()
   *  @param val   Default value for this argument
   */
  AppCmdArgList(AppCmdLine& parser, const std::string& name, const std::string& descr, const container& val);

  // Destructor
  virtual ~AppCmdArgList( ) {}

  /**
   *  True if the value of the option was changed from command line. Only
   *  makes sense for "optional arguments", for required this will always
   *  return true.
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

  /**
   *  Return default value of the argument
   */
  const container& defValue() const { return _defValue ; }

protected:

  /**
   *  Is it required?
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
  virtual size_t maxWords () const { return ULONG_MAX ; }

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
  const bool _required ;
  const container _defValue ;
  container _value ;
  bool _changed ;

  // This class is non-copyable
  AppCmdArgList( const AppCmdArgList& );
  AppCmdArgList& operator= ( const AppCmdArgList& );

};

//  Make a required positional argument
template <typename Type>
AppCmdArgList<Type>::AppCmdArgList ( const std::string& name, const std::string& descr )
  : AppCmdArgBase()
  , _name(name)
  , _descr(descr)
  , _required(true)
  , _defValue()
  , _value()
  , _changed(false)
{
}

//  Make an optional positional argument
template <typename Type>
AppCmdArgList<Type>::AppCmdArgList ( const std::string& name, const std::string& descr, const container& val )
  : AppCmdArgBase()
  , _name(name)
  , _descr(descr)
  , _required(false)
  , _defValue(val)
  , _value(val)
  , _changed(false)
{
}

//  Make a required positional argument
template <typename Type>
AppCmdArgList<Type>::AppCmdArgList(AppCmdLine& parser, const std::string& name, const std::string& descr)
  : AppCmdArgBase()
  , _name(name)
  , _descr(descr)
  , _required(true)
  , _defValue()
  , _value()
  , _changed(false)
{
  parser.addArgument(*this);
}

//  Make an optional positional argument
template <typename Type>
AppCmdArgList<Type>::AppCmdArgList(AppCmdLine& parser, const std::string& name, const std::string& descr, const container& val)
  : AppCmdArgBase()
  , _name(name)
  , _descr(descr)
  , _required(false)
  , _defValue(val)
  , _value(val)
  , _changed(false)
{
  parser.addArgument(*this);
}

//  Set the value of the argument.
template <typename Type>
int
AppCmdArgList<Type>::setValue ( StringList::const_iterator begin,
                                StringList::const_iterator end )
{
  // sequence must be non-empty
  assert ( begin != end ) ;

  container localCont ;

  for ( ; begin != end ; ++ begin ) {
    // this may throw
    Type res = AppCmdTypeTraits<Type>::fromString ( *begin ) ;
    localCont.push_back ( res ) ;
  }

  _value = localCont ;
  _changed = true ;

  return _value.size() ;
}

} // namespace AppUtils

#endif  // APPUTILS_APPCMDARGLIST_HH
