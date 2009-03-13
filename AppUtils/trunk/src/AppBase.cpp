//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppBase...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "AppUtils/AppBase.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdExceptions.h"
#include "MsgLogger/MsgLogger.h"
#include "MsgLogger/MsgFormatter.h"
#include "MsgLogger/MsgHandlerStdStreams.h"
#include "MsgLogger/MsgLogLevel.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  /**
   *  removes dirname from the path name
   */
  std::string fixAppName ( const std::string& appName )
  {
    std::string::size_type idx = appName.rfind('/') ;
    if ( idx != std::string::npos ) {
      return std::string ( appName, idx+1 ) ;
    } else {
      return appName ;
    }
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace AppUtils {

//----------------
// Constructors --
//----------------
AppBase::AppBase ( const std::string& appName )
  : _cmdline( ::fixAppName(appName) )
  , _optVerbose( 'v', "verbose", "verbose output, multiple allowed", 0 )
  , _optQuiet( 'q', "quiet", "quieter output, multiple allowed", 2 )
{
  addOption( _optVerbose );
  addOption( _optQuiet );
}

//--------------
// Destructor --
//--------------
AppBase::~AppBase ()
{
}

/**
 *  Run the application
 */
int
AppBase::run ( int argc, char** argv )
{
  // parse command line, set all options and arguments
  try {
    _cmdline.parse ( argc, argv ) ;
  } catch ( AppCmdException& e ) {
    std::cerr << "Error parsing command line: " << e.what() << "\n"
              << "Use -h or --help option to obtain usage information" << std::endl ;
    return 2 ;
  }

  if ( _cmdline.helpWanted() ) {
    _cmdline.usage( std::cout ) ;
    this->moreUsage ( std::cout ) ;
    return 0 ;
  }

  // setup message logger
  MsgLogger::MsgLogLevel loglvl ( _optQuiet.value() - _optVerbose.value() ) ;
  MsgLogger::MsgLogger rootlogger ;
  rootlogger.setLevel ( loglvl ) ;

  // Do some smart formatting of the messages
  const char* fmt = "[%(LVL)] %(message)" ;
  const char* errfmt = "[%(LVL)] (%(time)) %(file):%(line) - %(message)" ;
  const char* dbgfmt = errfmt ;
  MsgLogger::MsgFormatter::addGlobalFormat ( fmt ) ;
  MsgLogger::MsgFormatter::addGlobalFormat ( MsgLogger::MsgLogLevel::debug, dbgfmt ) ;
  MsgLogger::MsgFormatter::addGlobalFormat ( MsgLogger::MsgLogLevel::trace, dbgfmt ) ;
  MsgLogger::MsgFormatter::addGlobalFormat ( MsgLogger::MsgLogLevel::warning, errfmt ) ;
  MsgLogger::MsgFormatter::addGlobalFormat ( MsgLogger::MsgLogLevel::error, errfmt ) ;
  MsgLogger::MsgFormatter::addGlobalFormat ( MsgLogger::MsgLogLevel::fatal, errfmt ) ;

  // pre-run
  try {
    int stat = preRunApp() ;
    if ( stat != 0 ) return stat ;
  } catch ( std::exception& e ) {
    std::cerr << "Standard exception caught in preRunApp(): " << e.what() << std::endl ;
    return 2 ;
  } catch ( ... ) {
    std::cerr << "Unknown exception caught in preRunApp()" << std::endl ;
    return 2 ;
  }

  // call subclass for some real stuff
  try {
    int stat = this->runApp() ;
    if ( stat != 0 ) return stat ;
  } catch ( std::exception& e ) {
    std::cerr << "Standard exception caught in runApp(): " << e.what() << std::endl ;
    return 2 ;
  } catch ( ... ) {
    std::cerr << "Unknown exception caught in runApp()" << std::endl ;
    return 2 ;
  }

  // clean-up
  try {
    int stat = postRunApp() ;
    if ( stat != 0 ) return stat ;
  } catch ( std::exception& e ) {
    std::cerr << "Standard exception caught in postRunApp(): " << e.what() << std::endl ;
    return 2 ;
  } catch ( ... ) {
    std::cerr << "Unknown exception caught in postRunApp()" << std::endl ;
    return 2 ;
  }

  return 0 ;
}

/**
 *  add command line option or argument, typically called from subclass constructor
 */
void
AppBase::setOptionsFile ( AppCmdOpt<std::string>& option )
{
  _cmdline.setOptionsFile ( option ) ;
}

void
AppBase::addOption ( AppCmdOptBase& option )
{
  _cmdline.addOption ( option ) ;
}

void
AppBase::addArgument ( AppCmdArgBase& arg )
{
  _cmdline.addArgument ( arg ) ;
}

/**
 *  print some additional info after the usage information is printed.
 */
void
AppBase::moreUsage ( std::ostream& out ) const
{
}

/**
 *  Method called before runApp, can be overridden in subclasses.
 *  Usually if you override it, call base class method too.
 */
int
AppBase::preRunApp ()
{
  return 0 ;
}

/**
 *  Method called after runApp, can be overridden in subclasses.
 *  Usually if you override it, call base class method too.
 */
int
AppBase::postRunApp ()
{
  return 0 ;
}

} // namespace AppUtils
