//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigSvcImplFile...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigSvc/ConfigSvcImplFile.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>

using namespace std;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ConfigSvc/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ConfigSvc {

//----------------
// Constructors --
//----------------
ConfigSvcImplFile::ConfigSvcImplFile ()
  : ConfigSvcImplI()
  , m_config()
{
}

ConfigSvcImplFile::ConfigSvcImplFile (const string& file)
  : ConfigSvcImplI()
  , m_config()
{
  // copied from AppUtils/AppCmdLine.cpp
  
  // open the file
  ifstream istream ( file.c_str() ) ;
  if ( not istream ) {
    // failed to open file
    throw ExceptionFileMissing ( file ) ;
  }

  readStream( istream, file );
}

ConfigSvcImplFile::ConfigSvcImplFile (istream& stream, const string& file)
  : ConfigSvcImplI()
  , m_config()
{
  readStream( stream, file );
}

//--------------
// Destructor --
//--------------
ConfigSvcImplFile::~ConfigSvcImplFile ()
{
}


// get the value of a single parameter, will throw ExceptionMissing 
// if parameter is not there
boost::shared_ptr<const string>
ConfigSvcImplFile::get(const string& section, 
                       const string& param) const
{
  SectionMap::const_iterator sitr = m_config.find(section);
  if ( sitr == m_config.end() ) return boost::shared_ptr<const string>();
  ParamMap::const_iterator pitr = sitr->second.find(param);
  if ( pitr == sitr->second.end() ) return boost::shared_ptr<const string>();
  return pitr->second;
}

// get a list of all sections
list<string>
ConfigSvcImplFile::getSections() const 
{
  list<string> list;
  SectionMap::const_iterator sitr;
  for (sitr = m_config.begin(); sitr != m_config.end(); sitr++) {
    list.push_back((*sitr).first);
  }
  return list;
}

// get a list of all parameters, or an empty list if the section is not found
list<string>
ConfigSvcImplFile::getKeys(const string& section) const 
{
  list<string> list;
  SectionMap::const_iterator sitr = m_config.find(section);
  if (sitr != m_config.end()) {
    ParamMap map = sitr->second;
    for (ParamMap::iterator pitr = map.begin(); pitr != map.end(); pitr++) {
      list.push_back((*pitr).first);
    }
  }
  return list;
}

// get the value of a single parameter as sequence, will throw ExceptionMissing
// if parameter is not there
boost::shared_ptr<const list<string> >
ConfigSvcImplFile::getList(const string& section,
                           const string& param) const
{
  // first look into the lists map
  ParamListMap& plmap = m_lists[section];
  ParamListMap::const_iterator plitr = plmap.find(param);
  if ( plitr != plmap.end() ) return plitr->second;

  // look into parameter map
  SectionMap::const_iterator sitr = m_config.find(section);
  if ( sitr == m_config.end() ) return boost::shared_ptr<const list<string> >();
  ParamMap::const_iterator pitr = sitr->second.find(param);
  if ( pitr == sitr->second.end() ) return boost::shared_ptr<const list<string> >();
  boost::shared_ptr<const string> line = pitr->second;
  
  // convert and add to the list map
  boost::shared_ptr<list<string> > l (new list<string>());
  plmap[param] = l;
  if (not line->empty()) {
    boost::split (*l, *line, boost::is_any_of(" \t"), boost::token_compress_on);
  }
  return l;
}

// set the value of the parameter, if parameter already exists it will be replaced
void 
ConfigSvcImplFile::put(const string& section, 
                       const string& param, 
                       const string& value)
{
  // remove cached list value if any
  m_lists[section].erase(param);
  
  // add to map
  m_config[section][param].reset(new string(value));
}

static string
trim(const string &s)
{
  size_t start = s.find_first_not_of(" \t\r\n");
  if (start == string::npos) {
    return "";
  }
  size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

// read input file from stream
void 
ConfigSvcImplFile::readStream(istream& in, const string& name)
{
  // read all the lines from the file
  string line ;
  string curline ;
  string section ;
  unsigned int nlines = 0 ;
  while ( getline ( in, curline ) ) {
    nlines ++ ;
    curline = trim(curline);

    // skip empty lines and comments
    if (curline == "" || curline[0] == '#') {
      continue;
    }

    line += curline;
    
    if (line[line.size()-1] == '\\') {
      // continuation, read next line and add to current one
      line.erase(line.size()-1); // remove the continuation character
      //cout << "line " << nlines << ": continuation \"" << line << "\"\n" ;      
      continue;
    }
    
    if (line[0] == '[' ) {
      if (line[line.size() - 1] != ']') {
        // must be section name, whole string ends with ']' and we take 
        // everything between as section name
        throw ExceptionSyntax(name, nlines, "illegal section name format (missing ']')");
      }

      // don't need to keep whitespace
      section = trim(line.substr(1, line.size() - 2));
      m_config[section]; // create map entry
      
      //cout << "line " << nlines << ": section [" << section << "]\n" ;
      line.clear();
      continue;
    }

    // we get an option line, check that section name is defined
    if ( section.empty() ) {
      throw ExceptionSyntax(name, nlines, "parameter outside of section");
    }
    
    // must be option name followed by equal sign
    string::size_type eqpos = line.find("=") ;
    if ( eqpos == string::npos ) {
      throw ExceptionSyntax(name, nlines, "equal sign missing after option name");
    }
    string optname(trim(line.substr(0, eqpos)));
    if (optname == "") {
      throw ExceptionSyntax(name, nlines, "option name is missing");
    }
    string optval(trim(line.substr(eqpos + 1).c_str()));
    if (optval == "") {
      throw ExceptionSyntax(name, nlines, "option value is missing");
    }

    // set the option
    m_config[section][optname] = boost::shared_ptr<string>(new string(optval));
    //cout << "line " << nlines << ": '" << optname << "' = '" << optval << "'\n" ;

    line.clear();
  }

  // check the status of the file, must be at EOF
  if ( not in.eof() ) {
    throw ExceptionFileRead(name) ;
  }


}


} // namespace ConfigSvc
