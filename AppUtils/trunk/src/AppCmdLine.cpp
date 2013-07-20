//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
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
#include <fstream>
#include <iomanip>
#include <set>
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdArgBase.h"
#include "AppUtils/AppCmdExceptions.h"
#include "AppUtils/AppCmdOptBase.h"
#include "AppUtils/AppCmdOptList.h"
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
  , _optionsFile(0)
  , _argv()
  , _helpWanted(false)
  , _iter()
  , _nWordsLeft(0)
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
void
AppCmdLine::addArgument ( AppCmdArgBase& arg )
{
  // check some things first
  if ( ! _positionals.empty() ) {

    // cannot add required after non-required
    if ( arg.isRequired() && ! _positionals.back()->isRequired() ) {
      throw AppCmdArgOrderException ( arg.name() ) ;
    }
  }

  _positionals.push_back ( &arg ) ;

}

/**
 *  Add one more command option. The argument supplied is not copied,
 *  only its address is remembered. The lifetime of the argument should extend
 *  to the parse() method of this class.
 */
void
AppCmdLine::addOption ( AppCmdOptBase& option )
{
  // check maybe some wants to redefine help options?
  if ( ::isHelpOption(option.longOption()) ) {
    throw AppCmdOptReservedException ( option.longOption() ) ;
  }
  if ( ::isHelpOption(option.shortOption()) ) {
    throw AppCmdOptReservedException ( option.shortOption() ) ;
  }

  // check maybe some wants to duplicate options?
 if ( not option.longOption().empty() and findLongOpt ( option.longOption() ) ) {
   throw AppCmdOptDefinedException ( option.longOption() ) ;
 }
 if ( option.shortOption() != '\0' && findShortOpt ( option.shortOption() ) ) {
   throw AppCmdOptDefinedException ( option.shortOption() ) ;
 }

 // fine, remember it
  _options.push_back ( &option ) ;
}

/**
 *  Add option which will specify the name of the options file.
 *  Only one options file is allowed per parser, attempt to add one
 *  more will result in exception. The lifetime of the argument should
 *  extend to the parse() method of this class.
 */
void
AppCmdLine::setOptionsFile ( AppCmdOptList<std::string>& option )
{
  // second attempt will fail
  if ( _optionsFile ) {
    throw AppCmdException ( "options file option already defined, cannot re-define" ) ;
  }

  // define a regular option
  addOption ( option ) ;

  // remember it
  _optionsFile = &option ;
}

/**
 *  Parse function examines command line and sets the corresponding arguments.
 *  If it returns false then you should not expect anything, just exit.
 */
void
AppCmdLine::parse ( int argc, char* argv[] )
{
  _argv.clear() ;
  std::copy ( argv+1, argv+argc, std::back_inserter( _argv ) ) ;

  doParse() ;
}
void
AppCmdLine::parse ( int argc, const char* argv[] )
{
  _argv.clear() ;
  std::copy ( argv+1, argv+argc, std::back_inserter( _argv ) ) ;

  doParse() ;
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
AppCmdLine::usage ( std::ostream& out ) const
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

    size_t optLen = 12 ; // strlen("-h|-?|--help")
    size_t nameLen = 0 ;
    for ( OptionsList::const_iterator it = _options.begin() ; it != _options.end() ; ++ it ) {
      size_t moptLen = 0 ;
      if ( (*it)->shortOption() != '\0' ) {
        moptLen += 2 ; // '-x'
        if ( not (*it)->longOption().empty() ) moptLen += 1 ; // '|'
      }
      if ( not (*it)->longOption().empty() ) {
        moptLen += 2 + (*it)->longOption().size() ; // '|'
      }
      size_t thisNameLen = (*it)->name().size() ;
      if ( optLen < moptLen ) optLen = moptLen ;
      if ( nameLen < thisNameLen ) nameLen = thisNameLen ;
    }

    out << "  Available options:\n" ;
    out << "    {" << setw(optLen) << "-h|-?|--help" << "} "
	<< setw(nameLen) << "" << "  print help message\n" ;
    for ( OptionsList::const_iterator it = _options.begin() ; it != _options.end() ; ++ it ) {
      std::string fopt ;
      if ( (*it)->shortOption() != '\0' ) {
        fopt = "-" ;
        fopt += (*it)->shortOption() ;
        if ( not (*it)->longOption().empty() ) fopt += '|' ;
      }
      if ( not (*it)->longOption().empty() ) {
        fopt += "--" ;
        fopt += (*it)->longOption() ;
      }
      out << "    {" << setw(optLen) << fopt.c_str() << "} "
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

/**
 * Get the complete command line
 */
std::string
AppCmdLine::cmdline() const
{
  std::string cmdl = _argv0 ;
  for ( StringList::const_iterator i = _argv.begin() ; i != _argv.end() ; ++ i ) {
    const std::string& arg = *i ;
    if ( arg.find_first_of(" \t\n\"") != std::string::npos ) {
      cmdl += " '" ;
      cmdl += arg ;
      cmdl += "'" ;
    } else {
      cmdl += " " ;
      cmdl += arg ;
    }
  }
  return cmdl ;
}

/// real parsing happens in this method
void
AppCmdLine::doParse()
{
  _helpWanted = 0 ;

  // reset all options and arguments to their default values
  std::for_each ( _options.begin(), _options.end(), std::mem_fun(&AppCmdOptBase::reset) ) ;
  std::for_each ( _positionals.begin(), _positionals.end(), std::mem_fun(&AppCmdArgBase::reset) ) ;

  // get options from command line
  parseOptions() ;
  if ( _helpWanted ) {
    return ;
  }

  // get options from an options file if any
  parseOptionsFile() ;

  // get remaining args
  parseArgs() ;
}

/// parse options
void
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
        throw AppCmdOptUnknownException ( optname ) ;
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

      // now give it to option, this may throw
      option->setValue ( value ) ;

    } else if ( word.size() > 1 && word[0] == '-' ) {

      // should be short option
      if ( ::isHelpOption(word[1]) ) {
	_helpWanted = true ;
	return ;
      }

      // find option with this short name
      AppCmdOptBase* option = findShortOpt ( word[1] ) ;
      if ( ! option ) {
        throw AppCmdOptUnknownException( word[1] ) ;
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
	// this may throw
        option->setValue ( value ) ;

      } else {

	// option without argument, but the word may be collection of options, like -vvqs

        // this may throw (but should not)
        option->setValue ( "" ) ;
	for ( size_t i = 2 ; i < word.size() ; ++ i ) {
	  if ( ::isHelpOption(word[i]) ) {
	    _helpWanted = true ;
	    return ;
	  }
	  AppCmdOptBase* option = findShortOpt ( word[i] ) ;
	  if ( ! option ) {
	    throw AppCmdOptUnknownException ( word[i] ) ;
	  }
	  if ( option->hasArgument() ) {
	    // do not allow mixture
	    throw AppCmdException ( std::string("option with argument (-") +
	        std::string(1,option->shortOption()) +
                ") cannot be mixed with other options: " + word ) ;
	  }
	  // this may throw (but should not)
          option->setValue ( "" ) ;
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

}

/// parse options file
void
AppCmdLine::parseOptionsFile()
{
  if ( not _optionsFile ) return ;

  // build the list of options that were modified on the command line,
  // we do not want to change these again
  std::set<std::string> changedOptions ;
  for ( OptionsList::const_iterator it = _options.begin() ; it != _options.end() ; ++ it ) {
    if ( (*it)->valueChanged() ) {
      changedOptions.insert ( (*it)->longOption() ) ;
    }
  }

  typedef AppCmdOptList<std::string>::const_iterator OFIter ;
  for ( OFIter ofiter = _optionsFile->begin() ; ofiter != _optionsFile->end() ; ++ ofiter ) {

    // find the name of the options file
    std::string optFile = *ofiter ;
    if ( optFile.empty() ) {
      // no file name given
      return ;
    }

    // open the file
    std::ifstream istream ( optFile.c_str() ) ;
    if ( not istream ) {
      // failed to open file
      throw AppCmdException ( "failed to open options file: " + optFile ) ;
    }

    // read all the lines from the file
    std::string line ;
    unsigned int nlines = 0 ;
    while ( std::getline ( istream, line ) ) {
      nlines ++ ;

      // skip comments
      std::string::size_type fchar = line.find_first_not_of(" \t") ;
      if ( fchar == std::string::npos ) {
        // empty line
        //std::cout << "line " << nlines << ": empty\n" ;
        continue ;
      } else if ( line[fchar] == '#' ) {
        // comment
        //std::cout << "line " << nlines << ": comment\n" ;
        continue ;
      }

      // get option name
      std::string::size_type optend = line.find_first_of( " \t=", fchar ) ;
      std::string optname ( line, fchar, optend ) ;

      // find option with this long name
      AppCmdOptBase* option = findLongOpt ( optname ) ;
      if ( ! option ) {
        throw AppCmdException ( "Error parsing options file: option '" + optname + "' is unknown" ) ;
      }

      // if it was changed on command line do not change it again
      if ( changedOptions.find(optname) != changedOptions.end() ) {
        continue ;
      }

      //std::cout << "line " << nlines << ": option '" << optname << "'\n" ;

      // get option value if any
      std::string optval ;
      if ( optend != std::string::npos ) {
        std::string::size_type pos1 = line.find( '=', optend ) ;
        //std::cout << "line " << nlines << ": pos1 = " << pos1 << "\n" ;
        if ( pos1 != std::string::npos ) {
          pos1 = line.find_first_not_of(" \t",pos1+1) ;
          //std::cout << "line " << nlines << ": pos1 = " << pos1 << "\n" ;
          if ( pos1 != std::string::npos ) {
            std::string::size_type pos2 = line.find_last_not_of( " \t" ) ;
            //std::cout << "line " << nlines << ": pos2 = " << pos2 << "\n" ;
            if ( pos2 != std::string::npos ) {
              optval = std::string ( line, pos1, pos2-pos1+1 ) ;
            } else {
              optval = std::string ( line, pos1 ) ;
            }
            //std::cout << "line " << nlines << ": value '" << optval << "'\n" ;
          }
        }

      }

      // set the option
      option->setValue ( optval ) ;
      //std::cout << "line " << nlines << ": '" << optname << "' = '" << optval << "'\n" ;

    }

    // check the status of the file, must be at EOF
    if ( not istream.eof() ) {
      throw AppCmdException ( "failure when reading options file, at or around line " + boost::lexical_cast<std::string>(nlines) ) ;
    }

  }
}

/// parse arguments
void
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
        throw AppCmdException ( "missing positional required argument(s)" ) ;
      }
      return ;
    }

    // determine how many words we could give to next argument
    size_t nWordsToGive = 1 ;
    if ( (*it)->maxWords() > 1 ) {
      // but can get more
      if ( _nWordsLeft <= nPosLeft ) {
	// too few words left
        throw AppCmdException ( "missing positional required argument(s)" ) ;
      }
      nWordsToGive = _nWordsLeft - nPosLeft ;
    }

    StringList::const_iterator w_end = _iter ;
    std::advance ( w_end, nWordsToGive ) ;
    // this can throw
    int consumed = (*it)->setValue ( _iter, w_end ) ;
    std::advance ( _iter, consumed ) ;
    _nWordsLeft -= consumed ;

  }

  if ( _iter != _argv.end() ) {
    // not whole line is consumed
    throw AppCmdException ( "command line is too long" ) ;
  }

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
