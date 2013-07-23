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
#include "AppUtils/AppCmdOptBool.h"
#include "AppUtils/AppCmdOptList.h"
#include "AppUtils/AppCmdWordWrap.h"
using std::ios;
using std::ostream;
using std::setw;

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

bool
isHelpOption(const std::string& optname)
{
  return optname == "help" or optname == "h" or optname == "?";
}

// special help option used internally
AppUtils::AppCmdOptBool helpOpt("h,?,help", "print help message");

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace AppUtils {

/*
 *  Constructor.
 */
AppCmdLine::AppCmdLine(const std::string& argv0)
    : AppCmdOptGroup("General options")
    , _groups()
    , _positionals()
    , _argv0(argv0)
    , _optionsFile(0)
    , _argv()
    , _helpWanted(false)
    , _iter()
    , _nWordsLeft(0)
{
  this->addOption(::helpOpt);
}

// Destructor
AppCmdLine::~AppCmdLine()
{
}

// Add options group to the parser.
void
AppCmdLine::addGroup(AppCmdOptGroup& group)
{
  _groups.push_back(&group);
}

/*
 *  Add one more positional argument. The argument supplied is not copied,
 *  only its address is remembered. The lifetime of the argument should extend
 *  to the parse() method of this class.
 */
void
AppCmdLine::addArgument(AppCmdArgBase& arg)
{
  _positionals.push_back(&arg);
}

/*
 *  Add option which will specify the name of the options file.
 *  Only one options file is allowed per parser, attempt to add one
 *  more will result in exception. The lifetime of the argument should
 *  extend to the parse() method of this class.
 */
void
AppCmdLine::setOptionsFile(AppCmdOptList<std::string>& option)
{
  // second attempt will fail
  if (_optionsFile) {
    throw AppCmdException("options file option already defined, cannot re-define");
  }

  // remember it
  _optionsFile = &option;
}

/*
 *  Parse function examines command line and sets the corresponding arguments.
 *  If it returns false then you should not expect anything, just exit.
 */
void
AppCmdLine::parse(int argc, char* argv[])
{
  _argv.clear();
  _argv.insert(_argv.end(), argv + 1, argv + argc);

  doParse();
}
void
AppCmdLine::parse(int argc, const char* argv[])
{
  _argv.clear();
  _argv.insert(_argv.end(), argv + 1, argv + argc);

  doParse();
}

/*
 *  Returns true if the "help" option was specified on the command line.
 *  Always check its return value after calling parse() when it returns true,
 *  if the help option is given then parse() stops without checking anything else.
 */
bool
AppCmdLine::helpWanted() const
{
  return _helpWanted;
}

/*
 *  Prints usage information
 */
void
AppCmdLine::usage(std::ostream& out) const
{
  out.setf(ios::left, ios::adjustfield);

  // build full list of options from all groups
  OptionsList options = this->options();
  for (GroupsList::const_iterator git = _groups.begin(); git != _groups.end(); ++ git) {
    const OptionsList& groupOptions = (*git)->options();
    options.insert(options.end(), groupOptions.begin(), groupOptions.end());
  }

  out << "\nUsage: " << _argv0;
  if (!options.empty()) {
    out << " [options]";
  }
  for (PositionalsList::const_iterator it = _positionals.begin(); it != _positionals.end(); ++it) {
    bool required = (*it)->isRequired();
    out << (required ? " " : " [") << (*it)->name() << ((*it)->maxWords() > 1 ? " ..." : "") << (required ? "" : "]");
  }
  out << '\n';

  if (!options.empty()) {

    // calculate max length of the options strings (-o|--option)
    size_t optLen = 0;
    size_t nameLen = 0;
    for (OptionsList::const_iterator it = options.begin(); it != options.end(); ++it) {
      const std::vector<std::string>& optnames = (*it)->options();
      size_t moptLen = optnames.size() - 1;  // all separating '|'s
      for (std::vector<std::string>::const_iterator oit = optnames.begin(); oit != optnames.end(); ++ oit) {
        moptLen += oit->size() + 1; // '-x'
        if (oit->size() > 1) {
          moptLen += 1; // '--xxx'
        }
      }
      size_t thisNameLen = (*it)->name().size();
      if (optLen < moptLen) optLen = moptLen;
      if (nameLen < thisNameLen) nameLen = thisNameLen;
    }

    if (_groups.empty()) {
      formatOptGroup(out, "Available options", this->options(), optLen, nameLen);
    } else {
      formatOptGroup(out, this->groupName(), this->options(), optLen, nameLen);
      for (GroupsList::const_iterator git = _groups.begin(); git != _groups.end(); ++ git) {
        formatOptGroup(out, (*git)->groupName(), (*git)->options(), optLen, nameLen);
      }
    }
  }

  if (!_positionals.empty()) {

    size_t nameLen = 0;
    for (PositionalsList::const_iterator it = _positionals.begin(); it != _positionals.end(); ++it) {
      if (nameLen < (*it)->name().size()) nameLen = (*it)->name().size();
    }

    AppCmdWordWrap ww;
    const int width = ww.pageWidth();
    const int width2 = width/2;
    int descrLen = width - nameLen - 7; // "....name.-.description....."
    bool nl = false;
    if (descrLen < width2) {
      descrLen = width2;
      nl = true;
    }

    // wrap description, print it on separate lines
    out << "\n  Positional parameters:\n";
    for (PositionalsList::const_iterator it = _positionals.begin(); it != _positionals.end(); ++it) {

      out << "    " << setw(nameLen) << (*it)->name().c_str() << " - ";

      std::vector<std::string> descr = ww.wrap((*it)->description(), descrLen);
      for (std::vector<std::string>::const_iterator it = descr.begin(); it != descr.end(); ++ it) {
        if (it == descr.begin() and nl) out << '\n';
        if (it != descr.begin() or nl) out << setw(width-descrLen) << "";
        out << *it << '\n';
      }
    }
  }

  out << "\n";
  out.setf(ios::right, ios::adjustfield);
}

/**
 * Get the complete command line
 */
std::string
AppCmdLine::cmdline() const
{
  std::string cmdl = _argv0;
  for (StringList::const_iterator i = _argv.begin(); i != _argv.end(); ++i) {
    const std::string& arg = *i;
    if (arg.find_first_of(" \t\n\"") != std::string::npos) {
      cmdl += " '";
      cmdl += arg;
      cmdl += "'";
    } else {
      cmdl += " ";
      cmdl += arg;
    }
  }
  return cmdl;
}

/// real parsing happens in this method
void
AppCmdLine::doParse()
{
  _helpWanted = 0;

  // build full list of options from all groups
  OptionsList options = this->options();
  for (GroupsList::const_iterator git = _groups.begin(); git != _groups.end(); ++ git) {
    const OptionsList& groupOptions = (*git)->options();
    options.insert(options.end(), groupOptions.begin(), groupOptions.end());
  }

  // if options-file option was set but it was not added to any group add it now to the parser
  if (_optionsFile) {
    if (std::find(options.begin(), options.end(), _optionsFile) == options.end()) {
      // define a regular option
      this->addOption(*_optionsFile);
      options.push_back(_optionsFile);
    }
  }

  // check for option name conflicts
  std::set<std::string> allNames;
  for (OptionsList::const_iterator oit = options.begin(); oit != options.end(); ++ oit) {
    const std::vector<std::string>& optnames = (*oit)->options();
    for (std::vector<std::string>::const_iterator it = optnames.begin(); it != optnames.end(); ++it) {
      if (allNames.count(*it)) {
        throw AppCmdOptDefinedException(*it);
      }
      allNames.insert(*it);
    }
  }
  allNames.clear();


  // check for arguments order
  for (PositionalsList::const_iterator it = _positionals.begin(); it != _positionals.end(); ++ it) {

    if (it != _positionals.begin()) {

      AppCmdArgBase* arg = *it;
      AppCmdArgBase* argPrev = *(it-1);

      // cannot have required after non-required
      if (arg->isRequired() and not argPrev->isRequired()) {
        throw AppCmdArgOrderException(arg->name());
      }

    }
  }

  // reset all options and arguments to their default values
  std::for_each(options.begin(), options.end(), std::mem_fun(&AppCmdOptBase::reset));
  std::for_each(_positionals.begin(), _positionals.end(), std::mem_fun(&AppCmdArgBase::reset));

  // get options from command line
  parseOptions(options);
  if (_helpWanted) {
    return;
  }

  // get options from an options file if any
  parseOptionsFile(options);

  // get remaining args
  parseArgs();
}

/// parse options
void
AppCmdLine::parseOptions(const OptionsList& options)
{
  _iter = _argv.begin();
  _nWordsLeft = _argv.size();
  while (_iter != _argv.end()) {

    const std::string& word = *_iter;

    if (word == "--") {

      // should stop here
      ++_iter;
      break;

    } else if (word.size() > 2 && word[0] == '-' && word[1] == '-') {

      // long option takes everything before '='
      const std::string optname(word, 2, word.find('=') - 2);

      // long options should be longer than one character
      if (optname.size() < 2) {
        throw AppCmdOptUnknownException(optname);
      }

      // if --help is provided stop parsing
      if (::isHelpOption(optname)) {
        _helpWanted = true;
        break;
      }

      // find option with this name
      AppCmdOptBase* option = findOpt(optname, options);
      if (!option) {
        throw AppCmdOptUnknownException(optname);
      }

      // option argument value (only for options with arguments)
      std::string value;
      if (option->hasArgument()) {
        // take everything after the '=' or next word
        std::string::size_type eqpos = word.find('=');
        if (eqpos != std::string::npos) {
          value = std::string(word, eqpos + 1);
        } else {
          ++_iter;
          --_nWordsLeft;
          value = *_iter;
        }
      }

      // now give it to option, this may throw
      option->setValue(value);

    } else if (word.size() > 1 && word[0] == '-') {

      // should be short option or options
      std::string optname(1, word[1]);

      // stop on -h
      if (::isHelpOption(optname)) {
        _helpWanted = true;
        return;
      }

      // find option with this short name
      AppCmdOptBase* option = findOpt(optname, options);
      if (!option) {
        throw AppCmdOptUnknownException(optname);
      }

      if (option->hasArgument()) {

        // option expects argument, it is either the rest of this word or next word
        std::string value;
        if (word.size() == 2) {
          ++_iter;
          --_nWordsLeft;
          value = *_iter;
        } else {
          value = std::string(word, 2);
        }
        // this may throw
        option->setValue(value);

      } else {

        // option without argument, but the word may be collection of options, like -vvqs

        // this may throw (but should not)
        option->setValue("");

        // scan remaining characters which should all be single-char options with no argument
        for (size_t i = 2; i < word.size(); ++i) {
          std::string optname(1, word[i]);

          if (::isHelpOption(optname)) {
            _helpWanted = true;
            return;
          }
          AppCmdOptBase* option = findOpt(optname, options);
          if (!option) {
            throw AppCmdOptUnknownException(optname);
          }
          if (option->hasArgument()) {
            // do not allow mixture
            throw AppCmdException(
                std::string("option with argument (-") + optname
                    + ") cannot be mixed with other options: " + word);
          }
          // this may throw (but should not)
          option->setValue("");
        }

      }

    } else {

      // not an option, stop here. Note that '-' by itself is considered
      // as an argument.
      break;

    }

    ++_iter;
    --_nWordsLeft;

  }

}

/// parse options file
void
AppCmdLine::parseOptionsFile(const OptionsList& options)
{
  if (not _optionsFile) return;

  // build the list of options that were modified on the command line,
  // we do not want to change these again as command line overrides
  // options file contents.
  std::set < std::string > changedOptions;
  for (OptionsList::const_iterator it = options.begin(); it != options.end(); ++it) {
    if ((*it)->valueChanged()) {
      const std::vector<std::string>& optnames = (*it)->options();
      changedOptions.insert(optnames.begin(), optnames.end());
    }
  }

  typedef AppCmdOptList<std::string>::const_iterator OFIter;
  for (OFIter ofiter = _optionsFile->begin(); ofiter != _optionsFile->end(); ++ofiter) {

    // find the name of the options file
    std::string optFile = *ofiter;
    if (optFile.empty()) {
      // no file name given
      return;
    }

    // open the file
    std::ifstream istream(optFile.c_str());
    if (not istream) {
      // failed to open file
      throw AppCmdException("failed to open options file: " + optFile);
    }

    // read all the lines from the file
    std::string line;
    unsigned int nlines = 0;
    while (std::getline(istream, line)) {
      nlines++;

      // skip comments
      std::string::size_type fchar = line.find_first_not_of(" \t");
      if (fchar == std::string::npos) {
        // empty line
        //std::cout << "line " << nlines << ": empty\n" ;
        continue;
      } else if (line[fchar] == '#') {
        // comment
        //std::cout << "line " << nlines << ": comment\n" ;
        continue;
      }

      // get option name
      std::string::size_type optend = line.find_first_of(" \t=", fchar);
      std::string optname(line, fchar, optend);

      // find option with this long name
      AppCmdOptBase* option = findOpt(optname, options);
      if (!option) {
        throw AppCmdException("Error parsing options file: option '" + optname + "' is unknown");
      }

      // if it was changed on command line do not change it again
      if (changedOptions.find(optname) != changedOptions.end()) {
        continue;
      }

      //std::cout << "line " << nlines << ": option '" << optname << "'\n" ;

      // get option value if any
      std::string optval;
      if (optend != std::string::npos) {
        std::string::size_type pos1 = line.find('=', optend);
        //std::cout << "line " << nlines << ": pos1 = " << pos1 << "\n" ;
        if (pos1 != std::string::npos) {
          pos1 = line.find_first_not_of(" \t", pos1 + 1);
          //std::cout << "line " << nlines << ": pos1 = " << pos1 << "\n" ;
          if (pos1 != std::string::npos) {
            std::string::size_type pos2 = line.find_last_not_of(" \t");
            //std::cout << "line " << nlines << ": pos2 = " << pos2 << "\n" ;
            if (pos2 != std::string::npos) {
              optval = std::string(line, pos1, pos2 - pos1 + 1);
            } else {
              optval = std::string(line, pos1);
            }
            //std::cout << "line " << nlines << ": value '" << optval << "'\n" ;
          }
        }

      }

      // set the option
      option->setValue(optval);
      //std::cout << "line " << nlines << ": '" << optname << "' = '" << optval << "'\n" ;

    }

    // check the status of the file, must be at EOF
    if (not istream.eof()) {
      throw AppCmdException(
          "failure when reading options file, at or around line " + boost::lexical_cast < std::string > (nlines));
    }

  }
}

/// parse arguments
void
AppCmdLine::parseArgs()
{
  int nPosLeft = _positionals.size();
  for (PositionalsList::const_iterator it = _positionals.begin(); it != _positionals.end(); ++it) {

    // number of positional args left after the current one
    --nPosLeft;

    if (_iter == _argv.end()) {
      // no data left
      bool ok = !(*it)->isRequired();
      if (!ok) {
        throw AppCmdArgListTooShort();
      }
      return;
    }

    // determine how many words we could give to next argument
    size_t nWordsToGive = 1;
    if ((*it)->maxWords() > 1) {
      // but can get more
      if (_nWordsLeft <= nPosLeft) {
        // too few words left
        throw AppCmdArgListTooShort();
      }
      nWordsToGive = _nWordsLeft - nPosLeft;
    }

    StringList::const_iterator w_end = _iter;
    std::advance(w_end, nWordsToGive);
    // this can throw
    int consumed = (*it)->setValue(_iter, w_end);
    std::advance(_iter, consumed);
    _nWordsLeft -= consumed;

  }

  if (_iter != _argv.end()) {
    // not whole line is consumed
    throw AppCmdArgListTooLong();
  }

}

/// find option with the long name
AppCmdOptBase*
AppCmdLine::findOpt(const std::string& opt, const OptionsList& options) const
{
  for (OptionsList::const_iterator it = options.begin(); it != options.end(); ++it) {
    const std::vector<std::string>& optnames = (*it)->options();
    if (std::find(optnames.begin(), optnames.end(), opt) != optnames.end()) {
      return *it;
    }
  }
  return 0;
}

void
AppCmdLine::formatOptGroup(std::ostream& out, const std::string& groupName, const OptionsList& options, size_t optLen,
    size_t nameLen) const
{
  AppCmdWordWrap ww;
  const int width = ww.pageWidth();
  const int width2 = width/2;
  int descrLen = width - optLen - nameLen - 9; // "....{options}.name..description....."
  bool nl = false;
  if (descrLen < width2) {
    descrLen = width2;
    nl = true;
  }

  out << "\n  " << groupName << ":\n";
  for (OptionsList::const_iterator it = options.begin(); it != options.end(); ++it) {
    std::string fopt;

    // format all options
    const std::vector<std::string>& optnames = (*it)->options();
    for (std::vector<std::string>::const_iterator oit = optnames.begin(); oit != optnames.end(); ++ oit) {
      if (not fopt.empty()) fopt += "|";
      fopt += "-";
      if (oit->size() > 1) fopt += "-";
      fopt += *oit;
    }

    // wrap description, print it on separate lines
    std::vector<std::string> descr = ww.wrap((*it)->description(), descrLen);
    out << "    {" << setw(optLen) << fopt.c_str() << "} " << setw(nameLen) << (*it)->name().c_str() << "  ";
    for (std::vector<std::string>::const_iterator it = descr.begin(); it != descr.end(); ++ it) {
      if (it == descr.begin() and nl) out << '\n';
      if (it != descr.begin() or nl) out << setw(width-descrLen) << "";
      out << *it << '\n';
    }
  }
}

} // namespace AppUtils
