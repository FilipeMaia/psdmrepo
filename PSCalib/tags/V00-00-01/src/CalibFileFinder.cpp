//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibFileFinder...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSCalib/CalibFileFinder.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <list>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace fs = boost::filesystem;

namespace {

  const char logger[] = "CalibFileFinder";
  
  // helper class for ordering files in a directory
  class CalibFile {
  public:
    CalibFile(const fs::path& path) 
      : m_path(path)
    {
      std::string basename = path.stem();
      std::string::size_type p = basename.find('-');
      if (p == std::string::npos) { 
        throw std::runtime_error("missing dash in filename: " + path.string());
      }
      
      std::string beginstr(basename, 0, p);
      std::string endstr(basename, p+1);
      
      m_begin = boost::lexical_cast<unsigned>(beginstr);
      if (endstr == "end") {
        m_end = std::numeric_limits<unsigned>::max();
      } else {
        m_end = boost::lexical_cast<unsigned>(endstr);
      }      
    }
    
    const fs::path path() const { return m_path; }
    unsigned begin() const { return m_begin; }
    unsigned end() const { return m_end; }

    // comparison for sorting
    bool operator<(const CalibFile& other) const {
      if (m_begin != other.m_begin) return m_begin < other.m_begin;
      return m_end < other.m_end;
    }
    
  private:
    fs::path m_path;
    unsigned m_begin;
    unsigned m_end;
  };
  
  std::ostream& operator<<(std::ostream& out, const CalibFile& cf) {
    return out << "CalibFile(\"" << cf.path() << "\", " << cf.begin() << ", " << cf.end() << ")" ;
  }
  
}


//----------------------------------------
//-- Public Function Member Definitions --
//----------------------------------------

namespace PSCalib {

//----------------
// Constructors --
//----------------

CalibFileFinder::CalibFileFinder (const std::string& calibDir,           //  /reg/d/psdm/cxi/cxi35711/calib
                                  const std::string& typeGroupName,      //  CsPad::CalibV1
                                  const std::string& src)                //  CxiDs1.0:Cspad.0
  : m_calibDir(calibDir)
  , m_typeGroupName(typeGroupName)
  , m_src(src)
{
}

//----------------

// find calibration file
std::string 
CalibFileFinder::findCalibFile(const std::string& dataType, unsigned long& runNumber) const        //  pedestals
try {
  // if no directory given then don't do anything
  if ( m_calibDir == "" ) return std::string();
  
  // construct full path name
  fs::path dir = m_calibDir;
  dir /= m_typeGroupName;
  dir /= m_src;
  dir /= dataType;

  // scan directory
  std::list<CalibFile> files;
  typedef fs::directory_iterator dir_iter;
  for (dir_iter diriter = dir_iter(dir); diriter != dir_iter(); ++ diriter ) {
    
    const fs::path& path = diriter->path();
    
    // only take *.data files
    if (path.extension() != ".data") {
      MsgLog(logger, info, "skipping file: " + path.string());
      continue;
    }

    try {
      files.push_back(CalibFile(path));
    } catch (const std::exception& ex) {
      MsgLog(logger, warning, "skipping file: " + path.string() + ": " + ex.what());
    }
    
  }

  unsigned long run = runNumber;

  // find the last file in the list whose begin run is less or equal 
  // to the run number
  files.sort();
  typedef std::list<CalibFile>::const_reverse_iterator FileIter;
  for (FileIter it = files.rbegin() ; it != files.rend() ; ++ it ) {
    MsgLog(logger, debug, "trying: " << *it << " for run " << run);
    if (it->begin() <= run and run <= it->end()) return it->path().string();
  }
  return std::string();

} catch (const fs::filesystem_error& ex) {
  // means cannot read directory
  return std::string();  
}


//--------------
// Destructor --
//--------------
CalibFileFinder::~CalibFileFinder ()
{
}

} // namespace PSCalib
