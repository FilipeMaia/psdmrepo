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


/**
 *  This class is the command-line parser. Some ideas are borrowed from CLHEP
 *  but implementation is different, and CLHEP's bugs are fixed. Here is an
 *  example of its usage:
 *
 *  // instanciate parser
 *  AppCmdLine cmdline( argv[0] ) ;
 *
 *  bool ok ;
 *
 *  // add some options
 *  AppCmdOptIncr optVerbose ( 'v', "verbose", "produce more noise", 0 ) ;
 *  ok = cmdline.addOption ( optVerbose ) ;
 *  AppCmdOpt<BdbTime> optTime ( 'b', "begin", "time", "start time for interval scan", BdbTime::minusInfinity ) ;
 *  ok = cmdline.addOption ( optTime ) ;
 *
 *  // add some positional parameters, first is required, second is optional
 *  AppCmdArg<std::string> argString( "name", "specifies the name of the game" ) ;
 *  ok = cmdline.addArgument ( argString ) ;
 *  AppCmdArg<int> argInt( "retries", "optional number of retries, def: 1000", 1000 ) ;
 *  ok = cmdline.addArgument ( argInt ) ;
 *
 *  // parse command line, set all options and arguments
 *  ok = cmdline.parse ( argc-1, argv+1 ) ;
 *  if ( ! ok ) {
 *    cmdline.usage( std::cerr ) ;
 *    exit(2) ;  // exit() may not be good idea, for illustration only here
 *  } else if ( cmdline.helpWanted() ) {
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
 *  @version $Id: AppCmdLine.hh,v 1.3 2004/08/06 06:30:30 bartoldu Exp $
 *
 *  @author Andy Salnikov	(originator)
 */

namespace AppUtils {

class AppCmdArgBase ;
class AppCmdOptBase ;

class AppCmdLine {

public:

  /**
   *  Constructor takes the name of the command, this is typically argv[0].
   */
  AppCmdLine( const std::string& argv0 ) ;

  /// Destructor
  virtual ~AppCmdLine( );

  /**
   *  Add one more positional argument. The argument object supplied is not copied,
   *  only its address is remembered. The lifetime of the argument should extend
   *  to the parse() method of this class.
   */
  virtual bool addArgument ( AppCmdArgBase& arg ) ;

  /**
   *  Add one more command option. The option object supplied is not copied,
   *  only its address is remembered. The lifetime of the argument should extend
   *  to the parse() method of this class.
   */
  virtual bool addOption ( AppCmdOptBase& option ) ;

  /**
   *  Parse function examines command line and sets the corresponding arguments.
   *  If it returns false then you should not expect anything, just exit.
   */
  bool parse ( int argc, char* argv[] ) ;

  /**
   *  Another form of parse, takes iterators. Dereferencing iterators should
   *  produce something convertible to std::string.
   */
  template <typename Iter>
  bool parse ( Iter begin, Iter end ) ;

  /**
   *  Get the error string from the last parse operation. Makes sense only
   *  if parse() returned false.
   */
  const std::string& getErrorString() const { return _errString ; }

  /**
   *  Returns true if the "help" option was specified on the command line.
   *  Always check its return value after calling parse() when it returns true,
   *  if the help option is given then parse() stops without checking anything else.
   */
  bool helpWanted() const ;

  /**
   *  Prints usage information to specified stream.
   */
  virtual void usage ( std::ostream& out ) const ;

protected:

  // types
  typedef std::list< std::string > StringList ;
  typedef std::list< AppCmdArgBase* > PositionalsList ;
  typedef std::list< AppCmdOptBase* > OptionsList ;

  /// real parsing happens in this method
  virtual bool doParse() ;

  /// parse options
  virtual bool parseOptions() ;

  /// parse arguments
  virtual bool parseArgs() ;

  /// find option with the long name
  AppCmdOptBase* findLongOpt ( const std::string& opt ) const ;

  /// find option with the short name
  AppCmdOptBase* findShortOpt ( char opt ) const ;

private:

  // Friends

  // Data members
  OptionsList _options ;
  PositionalsList _positionals ;

  std::string _argv0 ;
  StringList _argv ;

  bool _helpWanted ;
  StringList::const_iterator _iter ; // iterator used by doParse()
  int _nWordsLeft ;                  // number of words not yet seen

  std::string _errString ;

  // Note: if your class needs a copy constructor or an assignment operator,
  //  make one of the following public and implement it.
  AppCmdLine( const AppCmdLine& );                // Copy Constructor
  AppCmdLine& operator= ( const AppCmdLine& );    // Assignment op


};


// have to put templated stuff here
template <typename Iter>
bool
AppCmdLine::parse( Iter begin, Iter end )
{
  _argv.clear() ;
  std::copy ( begin, end, std::back_inserter( _argv ) ) ;
  return doParse() ;
}

} // namespace AppUtils


#endif // APPUTILS_APPCMDLINE_HH
