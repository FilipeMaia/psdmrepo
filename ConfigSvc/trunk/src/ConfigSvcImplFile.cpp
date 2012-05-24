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

ConfigSvcImplFile::ConfigSvcImplFile (const std::string& file)
  : ConfigSvcImplI()
  , m_config()
{
  // copied from AppUtils/AppCmdLine.cpp
  
  // open the file
  std::ifstream istream ( file.c_str() ) ;
  if ( not istream ) {
    // failed to open file
    throw ExceptionFileMissing ( file ) ;
  }

  readStream( istream, file );
}

ConfigSvcImplFile::ConfigSvcImplFile (std::istream& stream, const std::string& file)
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
boost::shared_ptr<const std::string>
ConfigSvcImplFile::get(const std::string& section, 
                       const std::string& param) const
{
  SectionMap::const_iterator sitr = m_config.find(section);
  if ( sitr == m_config.end() ) return boost::shared_ptr<const std::string>();
  ParamMap::const_iterator pitr = sitr->second.find(param);
  if ( pitr == sitr->second.end() ) return boost::shared_ptr<const std::string>();
  return pitr->second;
}

// Get a list of all parameters, or an empty list if the section is not found.
std::list<std::string>
ConfigSvcImplFile::getKeys(const std::string& section) const 
{
  std::list<std::string> list;
  SectionMap::const_iterator sitr = m_config.find(section);
  if ( sitr == m_config.end() ) return list;
  ParamMap map = sitr->second;
  for (ParamMap::iterator it = map.begin(); it != map.end(); it++) {
    list.push_back((*it).first);
  }
  return list;
}

// get the value of a single parameter as sequence, will throw ExceptionMissing
// if parameter is not there
boost::shared_ptr<const std::list<std::string> >
ConfigSvcImplFile::getList(const std::string& section,
                           const std::string& param) const
{
  // first look into the lists map
  ParamListMap& plmap = m_lists[section];
  ParamListMap::const_iterator plitr = plmap.find(param);
  if ( plitr != plmap.end() ) return plitr->second;

  // look into parameter map
  SectionMap::const_iterator sitr = m_config.find(section);
  if ( sitr == m_config.end() ) return boost::shared_ptr<const std::list<std::string> >();
  ParamMap::const_iterator pitr = sitr->second.find(param);
  if ( pitr == sitr->second.end() ) return boost::shared_ptr<const std::list<std::string> >();
  boost::shared_ptr<const std::string> line = pitr->second;
  
  // convert and add to the list map
  boost::shared_ptr<std::list<std::string> > list (new std::list<std::string>());
  plmap[param] = list;
  if (not line->empty()) {
    boost::split (*list, *line, boost::is_any_of(" \t"), boost::token_compress_on);
  }
  return list;
}

// set the value of the parameter, if parameter already exists it will be replaced
void 
ConfigSvcImplFile::put(const std::string& section, 
                       const std::string& param, 
                       const std::string& value)
{
  // remove cached list value if any
  m_lists[section].erase(param);
  
  // add to map
  m_config[section][param].reset(new std::string(value));
}

// read input file from stream
void 
ConfigSvcImplFile::readStream(std::istream& in, const std::string& name)
{
  // read all the lines from the file
  std::string line ;
  std::string curline ;
  std::string section ;
  unsigned int nlines = 0 ;
  while ( std::getline ( in, curline ) ) {
    nlines ++ ;

    // skip comments
    std::string::size_type fchar = curline.find_first_not_of(" \t") ;
    if ( fchar == std::string::npos ) {
      // empty line
      //std::cout << "line " << nlines << ": empty\n" ;
      curline.clear();
    } else if ( curline[fchar] == '#' ) {
      // comment
      //std::cout << "line " << nlines << ": comment\n" ;
      continue;
    }

    line += curline;
    
    // skip empty lines
    if (line.empty()) {
      //std::cout << "line " << nlines << ": empty\n" ;
      continue;
    }
    
    if (line[line.size()-1] == '\\') {
      // continuation, read next line and add to current one
      line.erase(line.size()-1);
      //std::cout << "line " << nlines << ": continuation \"" << line << "\"\n" ;      
      continue;
    }
    
    fchar = line.find_first_not_of(" \t") ;
    if ( line[fchar] == '[' ) {
      // must be section name, whole string ends with ']' and we take 
      // everything between as section name
      std::string::size_type lchar = line.find_last_not_of(" \t\r") ;
      if ( lchar == std::string::npos or line[lchar] != ']' ) {
        throw ExceptionSyntax(name, nlines, "illegal section name format");
      }

      // don't need to keep whitespace
      fchar = line.find_first_not_of(" \t", fchar+1) ;
      lchar = line.find_last_not_of(" \t", lchar-1) ;
      section = line.substr( fchar, lchar-fchar+1 );
      m_config[section];
      
      //std::cout << "line " << nlines << ": section [" << section << "]\n" ;
      line.clear();
      continue;
    }
    
    // we get an option line, check that section name is defined
    if ( section.empty() ) {
      throw ExceptionSyntax(name, nlines, "parameter outside of section");
    }
    
    // must be option name followed by equal sign
    std::string::size_type eqpos = line.find( "=", fchar ) ;
    if ( eqpos == std::string::npos ) {
      throw ExceptionSyntax(name, nlines, "equal sign missing after option name");
    }
    if ( eqpos == fchar ) {
      throw ExceptionSyntax(name, nlines, "option name is missing");
    }
    
    std::string::size_type optend = line.find_last_not_of( " \t", eqpos-1 ) ;
    std::string optname ( line, fchar, optend-fchar+1 ) ;

    //std::cout << "line " << nlines << ": option '" << optname << "'\n" ;

    // get option value
    std::string optval ;
    std::string::size_type pos1 = line.find_first_not_of(" \t",eqpos+1) ;
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

    // set the option
    m_config[section][optname] = boost::shared_ptr<std::string>(new std::string(optval));
    //std::cout << "line " << nlines << ": '" << optname << "' = '" << optval << "'\n" ;

    line.clear();
  }

  // check the status of the file, must be at EOF
  if ( not in.eof() ) {
    throw ExceptionFileRead(name) ;
  }


}


} // namespace ConfigSvc
