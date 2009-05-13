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

//-------------
// C Headers --
//-------------
extern "C" {
}

//---------------
// C++ Headers --
//---------------
#include <list>

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

namespace AppUtils {

template<class Type>
class AppCmdArgList : public AppCmdArgBase {

public:

  typedef std::list<Type> container ;
  typedef typename container::const_iterator const_iterator ;
  typedef typename container::size_type size_type ;

  /**
   *  Make a required positional argument
   */
  AppCmdArgList ( const std::string& name, const std::string& descr ) ;

  /**
   *  Make an optional positional argument
   */
  AppCmdArgList ( const std::string& name, const std::string& descr, const container& val ) ;

  // Destructor
  virtual ~AppCmdArgList( ) throw() ;

  /**
   *  Is it required?
   */
  virtual bool isRequired() const throw() ;

  /**
   *  Get the name of the paramater
   */
  virtual const std::string& name() const throw() ;

  /**
   *  Get one-line description
   */
  virtual const std::string& description() const throw() ;

  /**
   *  How many words from command line could this argument take? Single-word
   *  parameters should return 1. Parameters that can take the list of words
   *  Should return some big number. Note there is no function minWords() because
   *  it would always return 1.
   */
  virtual size_t maxWords () const throw() ;

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
                         StringList::const_iterator end ) throw(AppCmdException) ;

  /**
   *  True if the value of the option was changed from command line. Only
   *  makes sense for "optionsl arguments", for required this will always
   *  return true.
   */
  virtual bool valueChanged() const throw() ;

  /**
   *  Return current value of the argument
   */
  virtual const container& value() const throw() ;

  /**
   *  Return iterator to the begin/end of sequence
   */
  virtual const_iterator begin() const throw() ;
  virtual const_iterator end() const throw() ;

  /**
   *  Other usual container stuff
   */
  size_type size() const throw() { return _value.size() ; }
  bool empty() const throw() { return _value.empty() ; }

  /**
   *  Clear the collected values
   */
  virtual void clear() throw() ;

  /**
   *  Return default value of the argument
   */
  const container& defValue() const throw() { return _defValue ; }

  /**
   *  Reset argument to its default value
   */
  virtual void reset() throw() ;

protected:

  // Helper functions

private:

  // Friends

  // Data members
  const std::string _name ;
  const std::string _descr ;
  const bool _required ;
  const container _defValue ;
  container _value ;
  bool _changed ;

  // Note: if your class needs a copy constructor or an assignment operator,
  //  make one of the following public and implement it.
  AppCmdArgList( const AppCmdArgList<Type>& );  // Copy Constructor
  AppCmdArgList<Type>& operator= ( const AppCmdArgList<Type>& );

};

} // namespace AppUtils

#include  "AppUtils/AppCmdArgList.icc"

#endif  // APPUTILS_APPCMDARGLIST_HH
