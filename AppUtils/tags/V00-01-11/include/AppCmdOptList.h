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

#ifndef APPUTILS_APPCMDOPTLIST_HH
#define APPUTILS_APPCMDOPTLIST_HH

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

//		---------------------
// 		-- Class Interface --
//		---------------------


/**
 *  This class defines a command line option with argument. This is a templated
 *  class parameterized by the type of the argument. Any type supported by the
 *  AppCmdTypeTraits can be used as a template parameter.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2003		SLAC
 *
 *  @see AppCmdOptListBase
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov	(originator)
 */

namespace AppUtils {

template<class Type>
class AppCmdOptList : public AppCmdOptBase {

public:

  typedef std::list<Type> container ;
  typedef typename container::const_iterator const_iterator ;
  typedef typename container::size_type size_type ;

  /**
   *  Make an option with an argument
   */
  AppCmdOptList ( char shortOpt,
                  const std::string& longOpt,
                  const std::string& name,
                  const std::string& descr,
                  char separator = ',' ) ;
  // make option with long name only
  AppCmdOptList ( const std::string& longOpt,
                  const std::string& name,
                  const std::string& descr,
                  char separator = ',' ) ;
  // make option with short name only
  AppCmdOptList ( char shortOpt,
                  const std::string& name,
                  const std::string& descr,
                  char separator = ',' ) ;

  /// Destructor
  virtual ~AppCmdOptList( ) throw() ;

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
   *  Reset option to its default value, clear changed flag
   */
  virtual void reset() throw() ;

protected:

  // Helper functions

private:

  // Friends

  // Data members
  const char _shortOpt ;
  const std::string _longOpt ;
  const std::string _name ;
  const std::string _descr ;
  const char _separator ;
  container _value ;
  bool _changed ;

  // Note: if your class needs a copy constructor or an assignment operator,
  //  make one of the following public and implement it.
  AppCmdOptList( const AppCmdOptList<Type>& );  // Copy Constructor
  AppCmdOptList<Type>& operator= ( const AppCmdOptList<Type>& );

};

} // namespace AppUtils

#include  "AppUtils/AppCmdOptList.icc"


#endif  // APPUTILS_APPCMDOPTLIST_HH
