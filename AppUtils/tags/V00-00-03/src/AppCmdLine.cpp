//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: AppCmdLine.cc,v 1.6 2004/11/25 00:55:19 salnikov Exp $
//
// Description:
//	Class AppCmdLine
//
// Environment:
//	Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//	Andy Salnikov		originator
//
// Copyright Information:
//	Copyright (C) 2003	SLAC
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "AppUtils/AppCmdLine.h"

//-------------
// C Headers --
//-------------
extern "C" {
}

//---------------
// C++ Headers --
//---------------
#include <algorithm>
#include <functional>
#include <iterator>
#include <iomanip>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "AppUtils/AppCmdArgBase.h"
#include "AppUtils/AppCmdOptBase.h"
using std::ios;
using std::ostream;
using std::setw;

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  std::string longName ( const std::string& opt ) {
    // long option take everything before '='
    std::string::size_type rlen = opt.find ( '=' ) ;
    if ( rlen != std::string::npos ) {
      rlen -= 2 ;
    }
    return std::string ( opt, 2, rlen ) ;
  }

  bool isHelpOption ( char ch ) {
    return ch == 'h' || ch == '?' ;
  }

  bool isHelpOption ( const std::string& optname ) {
    return optname == "help" ;
  }


  const char* logger = "AppUtils.AppCmdLine" ;
}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace AppUtils {

/**
 *  Constructor.
 */
AppCmdLine::AppCmdLine( const std::string& argv0 )
  : _options()
  , _positionals()
  , _argv0(argv0)
  , _argv()
  , _helpWanted(false)
  , _iter()
  , _nWordsLeft(0)
  , _errString()
{
}

/// Destructor
AppCmdLine::~AppCmdLine( )
{
}

/**
 *  Add one more positional argument. The argument supplied is not copied,
 *  only its address is remembered. The lifetime of the argument should extend
 *  to the parse() method of this class.
 */
bool
AppCmdLine::addArgument ( AppCmdArgBase& arg )
{
  // check some things first
  if ( ! _positionals.empty() ) {

    // cannot add required after non-required
    if ( arg.isRequired() && ! _positionals.back()->isRequired() ) {
      MsgLog(logger, error, "AppCmdLine::addArgument - cannot add required argument after non-required.\n"
                    << " - while adding argument '" << arg.name() << "'" ) ;
      return false ;
    }
  }

  _positionals.push_back ( &arg ) ;

  return true ;
}

/**
 *  Add one more command option. The argument supplied is not copied,
 *  only its address is remembered. The lifetime of the argument should extend
 *  to the parse() method of this class.
 */
bool
AppCmdLine::addOption ( AppCmdOptBase& option )
{
  // check maybe some wants to redefine help options?
  if ( ::isHelpOption(option.longOption()) ) {
    MsgLog(logger, error,  "AppCmdLine::addOption: long option '--" << option.longOption() << "' is reserved" ) ;
    return false ;
  }
  if ( ::isHelpOption(option.shortOption()) ) {
    MsgLog(logger, error, "AppCmdLine::addOption: short option '-" << option.shortOption() << "' is reserved" ) ;
    return false ;
  }

  // check maybe some wants to duplicate options?
 if ( findLongOpt ( option.longOption() ) ) {
   MsgLog(logger, error, "AppCmdLine::addOption: long option '--" << option.longOption() << "' already defined" ) ;
   return false ;
 }
 if ( findShortOpt ( option.shortOption() ) ) {
   MsgLog(logger, error, "AppCmdLine::addOption: short option '-" << option.shortOption() << "' already defined" ) ;
   return false ;
 }

 // fine, remember it
  _options.push_back ( &option ) ;
  return true ;
}

/**
 *  Parse function examines command line and sets the corresponding arguments.
 *  If it returns false then you should not expect anything, just exit.
 */
bool
AppCmdLine::parse ( int argc, char* argv[] )
{
  _argv.clear() ;
  std::copy ( argv+1, argv+argc, std::back_inserter( _argv ) ) ;

  return doParse() ;
}
bool
AppCmdLine::parse ( int argc, const char* argv[] )
{
  _argv.clear() ;
  std::copy ( argv+1, argv+argc, std::back_inserter( _argv ) ) ;

  return doParse() ;
}

/**
 *  Returns true if the "help" option was specified on the command line.
 *  Always check its return value after calling parse() when it returns true,
 *  if the help option is given then parse() stops without checking anything else.
 */
bool
AppCmdLine::helpWanted() const
{
  return _helpWanted ;
}

/**
 *  Prints usage information
 */
void
AppCmdLine::usage ( ostream& out ) const
{
  out.setf ( ios::left, ios::adjustfield ) ;

  out << "Usage: " << _argv0 ;
  if ( ! _options.empty() ) {
    out << " [options]" ;
  }
  for ( PositionalsList::const_iterator it = _positionals.begin() ; it != _positionals.end() ; ++ it ) {
    bool required = (*it)->isRequired() ;
    out << ( required ? " " : " [" )
        << (*it)->name()
        << ( (*it)->maxWords() > 1 ? " ..." : "" )
        << ( required ? "" : "]" ) ;
  }
  out << '\n' ;


  if ( ! _options.empty() ) {

    size_t longLen = 7 ;
    size_t nameLen = 0 ;
    for ( OptionsList::const_iterator it = _options.begin() ; it != _options.end() ; ++ it ) {
      size_t thisLongLen = (*it)->longOption().size() ;
      size_t thisNameLen = (*it)->name().size() ;
      if ( longLen < thisLongLen ) longLen = thisLongLen ;
      if ( nameLen < thisNameLen ) nameLen = thisNameLen ;
    }

    out << "  Available options:\n" ;
    out << "    {-h|-?" << setw(longLen) << "|--help" << "} "
	<< setw(nameLen) << "" << "  print help message\n" ;
    for ( OptionsList::const_iterator it = _options.begin() ; it != _options.end() ; ++ it ) {
      out << "    {-" << (*it)->shortOption() << "|--"
          << setw(longLen) << (*it)->longOption().c_str() << "} "
	  << setw(nameLen) << (*it)->name().c_str() << "  "
	  << (*it)->description() << '\n' ;
    }
  }

  if ( ! _positionals.empty() ) {
    size_t nameLen = 0 ;
    for ( PositionalsList::const_iterator it = _positionals.begin() ; it != _positionals.end() ; ++ it ) {
      if (  nameLen < (*it)->name().size() ) nameLen = (*it)->name().size() ;
    }
    out << "  Positional parameters:\n" ;
    for ( PositionalsList::const_iterator it = _positionals.begin() ; it != _positionals.end() ; ++ it ) {
      out << "    " << setw(nameLen) << (*it)->name().c_str() << "  "
	  << (*it)->description() << '\n' ;
    }
  }

  out.setf ( ios::right, ios::adjustfield ) ;
}

/// real parsing happens in this method
bool
AppCmdLine::doParse()
{
  _errString = "" ;
  _helpWanted = 0 ;

  // reset all options and arguments to their default values
  std::for_each ( _options.begin(), _options.end(), std::mem_fun(&AppCmdOptBase::reset) ) ;
  std::for_each ( _positionals.begin(), _positionals.end(), std::mem_fun(&AppCmdArgBase::reset) ) ;

  if ( ! parseOptions() ) {
    return false ;
  }
  if ( _helpWanted ) {
    return true ;
  }
  return parseArgs() ;
}

/// parse options
bool
AppCmdLine::parseOptions()
{
  _iter = _argv.begin() ;
  _nWordsLeft = _argv.size() ;
  while ( _iter != _argv.end() ) {

    const std::string& word = *_iter ;

    if ( word == "--" ) {

      // should stop here
      ++ _iter ;
      break ;

    } else if ( word.size() > 2 && word[0] == '-' && word[1] == '-' ) {

      // long option take everything before '='
      const std::string& optname = ::longName( word ) ;

      if ( ::isHelpOption(optname) ) {
	_helpWanted = true ;
	break ;
      }

      // find option with this long name
      AppCmdOptBase* option = findLongOpt ( optname ) ;
      if ( ! option ) {
        _errString = "unknown option: --" ;
        _errString += optname ;
	return false ;
      }

      // option argument value (only for options with arguments)
      std::string value ;
      if ( option->hasArgument() ) {
	// take everything after the '=' or next word
        std::string::size_type eqpos = word.find ( '=' ) ;
	if ( eqpos != std::string::npos ) {
	  value = std::string ( word, eqpos+1 ) ;
	} else {
	  ++ _iter ;
	  -- _nWordsLeft ;
          value = *_iter ;
	}
      }

      // now give it to option
      if ( ! option->setValue ( value ) ) {
        _errString = "option --" ;
        _errString += optname ;
        _errString += " does not accept value " ;
        _errString += value ;
        return false ;
      }

    } else if ( word.size() > 1 && word[0] == '-' ) {

      // should be short option
      if ( ::isHelpOption(word[1]) ) {
	_helpWanted = true ;
	return true ;
      }

      // find option with this short name
      AppCmdOptBase* option = findShortOpt ( word[1] ) ;
      if ( ! option ) {
        _errString = "unknown option: -" ;
        _errString += word[1] ;
	return false ;
      }

      if ( option->hasArgument() ) {

        // option expects argument, it is either the rest of this word or next word
        std::string value ;
	if ( word.size() == 2 ) {
	  ++ _iter ;
	  -- _nWordsLeft ;
	  value = *_iter ;
	} else {
	  value = std::string ( word, 2 ) ;
	}
        if ( ! option->setValue ( value ) ) {
          _errString = "option -" ;
          _errString += option->shortOption() ;
          _errString += " does not accept value " ;
          _errString += value ;
          return false ;
        }

      } else {

	// option without argument, but the word may be collection of options, like -vvqs

        if ( ! option->setValue ( "" ) ) {
          _errString = "option -" ;
          _errString += option->shortOption() ;
          _errString += " cannot be set" ;
          return false ;
        }
	for ( size_t i = 2 ; i < word.size() ; ++ i ) {
	  if ( ::isHelpOption(word[i]) ) {
	    _helpWanted = true ;
	    return true ;
	  }
	  AppCmdOptBase* option = findShortOpt ( word[i] ) ;
	  if ( ! option ) {
            _errString = "unknown option: -" ;
            _errString += word[i] ;
	    return false ;
	  }
	  if ( option->hasArgument() ) {
	    // do not allow mixture
            _errString = "option with argument (-";
            _errString += option->shortOption() ;
            _errString += ") cannot be mixed with other options: " ;
            _errString += word ;
	    return false ;
	  }
          if ( ! option->setValue ( "" ) ) {
            _errString = "option -" ;
            _errString += option->shortOption() ;
            _errString += " cannot be set" ;
            return false ;
          }
	}

      }


    } else {

      // not an option, stop here. Note that '-' by itself is considered
      // as an argument.
      break ;

    }

    ++ _iter ;
    -- _nWordsLeft ;

  }

  return true ;
}

/// parse arguments
bool
AppCmdLine::parseArgs()
{
  int nPosLeft = _positionals.size() ;
  for ( PositionalsList::const_iterator it = _positionals.begin() ;
       it != _positionals.end() ;
       ++ it ) {

    // number of positional args left after the current one
    -- nPosLeft ;

    if ( _iter == _argv.end() ) {
      // no data left
      bool ok = ! (*it)->isRequired() ;
      if ( ! ok ) {
        _errString = "missing positional required argument(s)" ;
      }
      return ok ;
    }

    // determine how many words we could give to next argument
    size_t nWordsToGive = 1 ;
    if ( (*it)->maxWords() > 1 ) {
      // but can get more
      if ( _nWordsLeft <= nPosLeft ) {
	// too few words left
        _errString = "missing positional required argument(s)" ;
	return false ;
      }
      nWordsToGive = _nWordsLeft - nPosLeft ;
    }

    StringList::const_iterator w_end = _iter ;
    std::advance ( w_end, nWordsToGive ) ;
    int consumed = (*it)->setValue ( _iter, w_end ) ;
    if ( consumed <= 0 ) {
      // could not parse it
      _errString = "positional argument " ;
      _errString += (*it)->name() ;
      _errString += " does not accept value" ;
      return false ;
    }
    std::advance ( _iter, consumed ) ;
    _nWordsLeft -= consumed ;

  }

  if ( _iter != _argv.end() ) {
    // not whole line is consumed
    _errString = "command line is too long" ;
    return false ;
  }

  return true ;
}

/// find option with the long name
AppCmdOptBase*
AppCmdLine::findLongOpt ( const std::string& opt ) const
{
  for ( OptionsList::const_iterator it = _options.begin() ; it != _options.end() ; ++ it ) {
    if ( (*it)->longOption() == opt ) {
      return *it ;
    }
  }
  return 0 ;
}

/// find option with the short name
AppCmdOptBase*
AppCmdLine::findShortOpt ( char opt ) const
{
  for ( OptionsList::const_iterator it = _options.begin() ; it != _options.end() ; ++ it ) {
    if ( (*it)->shortOption() == opt ) {
      return *it ;
    }
  }
  return 0 ;
}

} // namespace AppUtils
