#ifndef APPUTILS_APPCMDLINE_HH
#define APPUTILS_APPCMDLINE_HH

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
#include <string>
#include <list>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cassert>

//----------------------
// Base Class Headers --
//----------------------

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

class AppCmdArgBase ;
class AppCmdOptBase ;
template <typename T> class AppCmdOptList ;

/// @addtogroup AppUtils

/**
 *  @ingroup AppUtils
 *
 *  This class is the command-line parser. Some ideas are borrowed from CLHEP
 *  but implementation is different, and CLHEP's bugs are fixed. Here is an
 *  example of its usage:
 *
 *  // instanciate parser
 *  AppCmdLine cmdline( argv[0] ) ;
 *
 *  // add some options
 *  AppCmdOptIncr optVerbose ( 'v', "verbose", "produce more noise", 0 ) ;
 *  cmdline.addOption ( optVerbose ) ;
 *  AppCmdOpt<BdbTime> optTime ( 'b', "begin", "time", "start time for interval scan", BdbTime::minusInfinity ) ;
 *  cmdline.addOption ( optTime ) ;
 *
 *  // add some positional parameters, first is required, second is optional
 *  AppCmdArg<std::string> argString( "name", "specifies the name of the game" ) ;
 *  cmdline.addArgument ( argString ) ;
 *  AppCmdArg<int> argInt( "retries", "optional number of retries, def: 1000", 1000 ) ;
 *  cmdline.addArgument ( argInt ) ;
 *
 *  // parse command line, set all options and arguments
 *  try {
 *    cmdline.parse ( argc-1, argv+1 ) ;
 *  } catch ( AppCmdException& e ) {
 *    std::cerr << "Error parsing command line: " << e.what() << "\n"
 *              << "Use -h or --help option to obtain usage information" << std::endl ;
 *    exit(2) ;  // exit() may not be good idea, for illustration only here
 *  }
 *  if ( cmdline.helpWanted() ) {
 *    cmdline.usage( std::cout ) ;
 *    exit(0) ;  // exit() may not be good idea, for illustration only here
 *  }
 *
 *  // using the values set by the parser
 *  if ( optVerbose.value() > 1 ) {
 *    std::cout << "Starting game " << argString.value() << std::endl ;
 *  }
 *  // etc.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2003		SLAC
 *
 *  @see AppCmdArg
 *  @see AppCmdArgList
 *  @see AppCmdOpt
 *  @see AppCmdOptToggle
 *  @see AppCmdOptIncr
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov	(originator)
 */

class AppCmdLine {

public:

  /**
   *  Constructor takes the name of the command, this is typically argv[0].
   */
  explicit AppCmdLine( const std::string& argv0 ) ;

  /// Destructor
  virtual ~AppCmdLine( );

  /**
   *  Add one more positional argument. The argument object supplied is not copied,
   *  only its address is remembered. The lifetime of the argument should extend
   *  to the parse() method of this class.
   */
  virtual void addArgument ( AppCmdArgBase& arg ) throw(std::exception) ;

  /**
   *  Add one more command option. The option object supplied is not copied,
   *  only its address is remembered. The lifetime of the argument should extend
   *  to the parse() method of this class.
   */
  virtual void addOption ( AppCmdOptBase& option ) throw(std::exception) ;

  /**
   *  Add option which will specify the names of the options files.
   *  Only one such option is allowed per parser, attempt to add one
   *  more will result in exception. The lifetime of the argument should
   *  extend to the parse() method of this class.
   */
  virtual void setOptionsFile ( AppCmdOptList<std::string>& option ) throw(std::exception) ;

  /**
   *  Parse function examines command line and sets the corresponding arguments.
   *  If it returns false then you should not expect anything, just exit.
   */
  void parse ( int argc, char* argv[] ) throw(std::exception) ;
  void parse ( int argc, const char* argv[] ) throw(std::exception) ;

  /**
   *  Another form of parse, takes iterators. Dereferencing iterators should
   *  produce something convertible to std::string.
   */
  template <typename Iter>
  void parse ( Iter begin, Iter end ) throw(std::exception) ;

  /**
   *  Returns true if the "help" option was specified on the command line.
   *  Always check its return value after calling parse() when it returns true,
   *  if the help option is given then parse() stops without checking anything else.
   */
  bool helpWanted() const throw() ;

  /**
   *  Prints usage information to specified stream.
   */
  virtual void usage ( std::ostream& out ) const ;

  /**
   * Get the complete command line
   */
  std::string cmdline() const ;

protected:

  // types
  typedef std::list< std::string > StringList ;
  typedef std::list< AppCmdArgBase* > PositionalsList ;
  typedef std::list< AppCmdOptBase* > OptionsList ;

  /// real parsing happens in this method
  virtual void doParse() throw(std::exception) ;

  /// parse options
  virtual void parseOptions() throw(std::exception) ;

  /// parse options file
  virtual void parseOptionsFile() throw(std::exception) ;

  /// parse arguments
  virtual void parseArgs() throw(std::exception) ;

  /// find option with the long name
  AppCmdOptBase* findLongOpt ( const std::string& opt ) const throw() ;

  /// find option with the short name
  AppCmdOptBase* findShortOpt ( char opt ) const throw() ;

private:

  // Friends

  // Data members
  OptionsList _options ;
  PositionalsList _positionals ;

  std::string _argv0 ;
  AppCmdOptList<std::string>* _optionsFile ;

  StringList _argv ;

  bool _helpWanted ;
  StringList::const_iterator _iter ; // iterator used by doParse()
  int _nWordsLeft ;                  // number of words not yet seen

  // Note: if your class needs a copy constructor or an assignment operator,
  //  make one of the following public and implement it.
  AppCmdLine( const AppCmdLine& );                // Copy Constructor
  AppCmdLine& operator= ( const AppCmdLine& );    // Assignment op


};


// have to put templated stuff here
template <typename Iter>
void
AppCmdLine::parse( Iter begin, Iter end ) throw(std::exception)
{
  _argv.clear() ;
  std::copy ( begin, end, std::back_inserter( _argv ) ) ;
  return doParse() ;
}

} // namespace AppUtils


#endif // APPUTILS_APPCMDLINE_HH
