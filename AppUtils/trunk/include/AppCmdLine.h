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

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace AppUtils {
class AppCmdArgBase ;
class AppCmdOptBase ;
template <typename T> class AppCmdOptList ;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace AppUtils {

/// @addtogroup AppUtils

/**
 *  @ingroup AppUtils
 *
 *  @brief Command-line parser.
 *
 *  This class is the command-line parser. Some ideas are borrowed from CLHEP
 *  but implementation is different, and CLHEP's bugs are fixed. Here is an
 *  example of its usage:
 *
 *  @code
 *  // instantiate parser, pass the name of the application
 *  AppCmdLine cmdline(argv[0]);
 *
 *  // add few options
 *  AppCmdOptIncr optVerbose("verbose,v", "produce more noise", 0);
 *  cmdline.addOption(optVerbose);
 *  AppCmdOpt<BdbTime> optTime ("begin,b", "time", "start time for interval scan", BdbTime::minusInfinity);
 *  cmdline.addOption(optTime);
 *
 *  // add two positional parameters, first is required, second is optional
 *  AppCmdArg<std::string> argString("name", "specifies the name of the game");
 *  cmdline.addArgument(argString);
 *  AppCmdArg<int> argInt("retries", "optional number of retries, def: 1000", 1000);
 *  cmdline.addArgument(argInt);
 *
 *  // parse command line, set all options and arguments
 *  try {
 *    cmdline.parse(argc, argv);
 *  } catch (const AppCmdException& e) {
 *    std::cerr << "Error parsing command line: " << e.what() << "\n"
 *              << "Use -h or --help option to obtain usage information" << std::endl;
 *    exit(2) ;  // exit() may not be good idea, for illustration only here
 *  }
 *  if (cmdline.helpWanted()) {
 *    cmdline.usage(std::cout);
 *    exit(0) ;  // exit() may not be good idea, for illustration only here
 *  }
 *
 *  // using the values set by the parser
 *  if (optVerbose.value() > 1) {
 *    std::cout << "Starting game " << argString.value() << std::endl;
 *  }
 *  // etc.
 *  @endcode
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
   *  @brief Make parser instance.
   *
   *  Constructor takes the name of the command, this is usually @c argv[0] but
   *  can be anything. This name is used in the output of usage() method to display
   *  usage line.
   */
  explicit AppCmdLine( const std::string& argv0 ) ;

  /// Destructor
  virtual ~AppCmdLine( );

  /**
   *  @brief Add one positional argument to parser.
   *
   *  The argument object supplied is not copied, only its address is remembered.
   *  The lifetime of the argument should extend to the parse() method of this class.
   *  This method may throw an exception if, for example, you try to add required
   *  argument after optional one.
   *
   *  @param[in] arg    Argument instance to add to the parser.
   *
   *  @throw AppCmdException or a subclass of it.
   */
  virtual void addArgument ( AppCmdArgBase& arg ) ;

  /**
   *  @brief Add one option to parser.
   *
   *  The option object supplied is not copied, only its address is remembered.
   *  The lifetime of the argument should extend to the parse() method of this class.
   *  This method may throw an exception if the option name conflicts with the previously
   *  added options.
   *
   *  @param[in] option   Option instance to add to the parser.
   *
   *  @throw AppCmdException or a subclass of it.
   */
  virtual void addOption ( AppCmdOptBase& option ) ;

  /**
   *  @brief Add option which specifies names of option files.
   *
   *  Add option which will specify the names of the options files.
   *  Only one such option is allowed per parser, attempt to add one
   *  more will result in exception. As the option type is AppCmdOptList
   *  it is possible to specify multiple files which will be read in the
   *  same order as they appear on command line. It is possible (but not
   *  required) to call addOption(option) first and then setOptionsFile(option)
   *  for the same option instance.
   *  The lifetime of the argument should extend to the parse() method of this class.
   *  This method may throw an exception if the option name conflicts with the previously
   *  added options.
   *
   *  @param[in] option   Option instance to add to the parser.
   *
   *  @throw AppCmdException or a subclass of it.
   */
  virtual void setOptionsFile ( AppCmdOptList<std::string>& option ) ;

  /**
   *  @brief Parse command line.
   *
   *  Parse method examines command line and sets the corresponding arguments.
   *  In case of errors (unknown options, conversion errors, etc.) it throws an
   *  exception of type AppCmdException.
   *
   *  Note that this method takes parameters that are passes to main() function,
   *  where argv[0] contains application name or path. parse() does not use
   *  argv[0] and discards its value internally so argv[0] can be any string,
   *  not necessary an application name.
   *
   *  @param[in] argc   Argument counter
   *  @param[in] argv   Argument vector
   *  @throw AppCmdException or any its subclass is thrown in case parsing fails.
   */
  void parse ( int argc, char* argv[] ) ;

  /**
   *  @brief Parse command line.
   *
   *  Parse method examines command line and sets the corresponding arguments.
   *  In case of errors (unknown options, conversion errors, etc.) it throws an
   *  exception of type AppCmdException.
   *
   *  Note that this method takes parameters that are passes to main() function,
   *  where argv[0] contains application name or path. parse() does not use
   *  argv[0] and discards its value internally so argv[0] can be any string,
   *  not necessary an application name.
   *
   *  @param[in] argc   Argument counter
   *  @param[in] argv   Argument vector
   *  @throw AppCmdException or any its subclass is thrown in case parsing fails.
   */
   void parse ( int argc, const char* argv[] ) ;

  /**
   *  @brief Parse command line.
   *
   *  Overloaded variant of parse() which accepts iterators. Value type for iterators
   *  should be type convertible to std::string. Unlike two other forms of parse()
   *  this method does not discard first element of sequence (*begin) so @c begin
   *  should point to first option or argument and not to application name. An example
   *  of its use with the @c argc and @c argv parameters passed to main():
   *  @code
   *  int main(int argc, char** argv) {
   *    AppCmdLine cmdline(argv[0]);
   *    // add few options here
   *    cmdline.parse(argv+1, argv+argc);
   *  }
   *  @endcode
   *
   *  @param[in] begin  Iterator pointing to first item.
   *  @param[in] end    Iterator pointing past the last item.
   *  @throw AppCmdException or any its subclass is thrown in case parsing fails.
   */
  template <typename Iter>
  void parse ( Iter begin, Iter end ) ;

  /**
   *  @brief Check whether -h or --help options were given.
   *
   *  Returns true if the "help" option was specified on the command line.
   *  Always check its return value after calling parse() when it returns true,
   *  if the help option is given then parse() stops without checking or parsing
   *  any other options or arguments.
   */
  bool helpWanted() const;

  /**
   *  @brief Print usage information to specified stream.
   */
  virtual void usage(std::ostream& out) const ;

  /**
   * @brief Get the complete command line as a string.
   *
   * Returns complete command line as a string, may be used for informational purposes.
   */
  std::string cmdline() const ;

protected:

  // types
  typedef std::vector< std::string > StringList ;
  typedef std::vector< AppCmdArgBase* > PositionalsList ;
  typedef std::vector< AppCmdOptBase* > OptionsList ;

  // real parsing happens in this method
  virtual void doParse() ;

  // parse options
  virtual void parseOptions() ;

  // parse options file
  virtual void parseOptionsFile() ;

  // parse arguments
  virtual void parseArgs() ;

  // find option with the long name
  AppCmdOptBase* findLongOpt ( const std::string& opt ) const ;

  // find option with the short name
  AppCmdOptBase* findShortOpt ( char opt ) const ;

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

  // This class in non-copyable
  AppCmdLine( const AppCmdLine& );
  AppCmdLine& operator= ( const AppCmdLine& );


};


// have to put templated stuff here
template <typename Iter>
void
AppCmdLine::parse( Iter begin, Iter end )
{
  _argv.clear() ;
  _argv.insert(_argv.end(), begin, end);
  return doParse() ;
}

} // namespace AppUtils


#endif // APPUTILS_APPCMDLINE_HH
