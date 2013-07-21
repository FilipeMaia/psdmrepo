#ifndef APPUTILS_APPCMDARGBASE_HH
#define APPUTILS_APPCMDARGBASE_HH

//--------------------------------------------------------------------------
//
// Environment:
//      This software was developed for the BaBar collaboration.  If you
//      use all or part of it, please give an appropriate acknowledgment.
//
// Copyright Information:
//      Copyright (C) 2003	SLAC
//
//------------------------------------------------------------------------

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdExceptions.h"

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
 *  @brief Base class for positional command line arguments.
 *
 *  This is an interface class for representation of the positional
 *  arguments in the command line.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgment.
 *
 *  Copyright (C) 2003		SLAC
 *
 *  @see AppCmdLine
 *  @see AppCmdArg
 *  @see AppCmdArgList
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov	(originator)
 */

class AppCmdArgBase {

public:

  typedef std::vector<std::string> StringList ;

  /// Destructor
  virtual ~AppCmdArgBase( ) ;

  /**
   *  True if the value of the option was changed from command line. Only
   *  makes sense for "optionsl arguments", for required this will always
   *  return true.
   */
  virtual bool valueChanged() const = 0 ;

protected:

  /**
   *  Constructor.
   */
  AppCmdArgBase() {}

  // The methods below are protected as they do need to be exposed
  // to user, this interface should only be used by AppCmdLine which
  // is declared as friend. Subclasses may use these methods as well
  // for implementing their own functionality or override them.

  /**
   *  Is it required?
   */
  virtual bool isRequired() const = 0 ;

  /**
   *  Get the name of the paramater
   */
  virtual const std::string& name() const = 0 ;

  /**
   *  Get one-line description
   */
  virtual const std::string& description() const = 0 ;

  /**
   *  How many words from command line could this argument take? Single-word
   *  parameters should return 1. Parameters that can take the list of words
   *  Should return some big number. Note there is no function minWords() because
   *  it would always return 1.
   */
  virtual size_t maxWords () const = 0 ;

  /**
   *  Set the value of the argument.
   *  Will throw an exception in case of conversion error.
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
                         StringList::const_iterator end ) = 0 ;

  /**
   *  Reset argument to its default value
   */
  virtual void reset() = 0 ;

private:

  // All private methods are accessible to the parser
  friend class AppCmdLine;

  // Data members

  // This class is non-copyable
  AppCmdArgBase( const AppCmdArgBase& );
  AppCmdArgBase& operator= ( const AppCmdArgBase& );


};

} // namespace AppUtils

#endif // APPUTILS_APPCMDARGBASE_HH
