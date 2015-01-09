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
#include <algorithm>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/DetInfo.hh"
#include "PSCalib/Exceptions.h"

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
      std::string basename = path.stem().string();
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
      return m_end > other.m_end;
    }
    
  private:
    fs::path m_path;
    unsigned m_begin;
    unsigned m_end;
  };
  
  std::ostream& operator<<(std::ostream& out, const CalibFile& cf) {
    return out << "CalibFile(\"" << cf.path() << "\", " << cf.begin() << ", " << cf.end() << ")" ;
  }
  

  // convert source address to string
  std::string toString( const Pds::Src& src )
  {
    if ( src.level() != Pds::Level::Source ) {
      throw PSCalib::NotDetInfoError(ERR_LOC);
    }

    const Pds::DetInfo& info = static_cast<const Pds::DetInfo&>( src ) ;
    std::ostringstream str ;
    str << Pds::DetInfo::name(info.detector()) << '.' << info.detId()
        << ':' << Pds::DetInfo::name(info.device()) << '.' << info.devId() ;
    return str.str() ;
  }

}


//----------------------------------------
//-- Public Function Member Definitions --
//----------------------------------------

namespace PSCalib {

//----------------
// Constructors --
//----------------

CalibFileFinder::CalibFileFinder (const std::string& calibDir,
                                  const std::string& typeGroupName,
                                  const unsigned&    print_bits)
  : m_calibDir(calibDir)
  , m_typeGroupName(typeGroupName)
  , m_print_bits(print_bits)
{
}

//--------------
// Destructor --
//--------------
CalibFileFinder::~CalibFileFinder ()
{
}

// find calibration file
std::string
CalibFileFinder::findCalibFile(const std::string& src, const std::string& dataType, unsigned long runNumber) const
try {
  // if no directory given then don't do anything
  if ( m_calibDir == "" ) return std::string();
  
  // construct full path name
  fs::path dir = m_calibDir;
  dir /= m_typeGroupName;
  dir /= src;
  dir /= dataType;

  // scan directory
  std::vector<std::string> files;
  typedef fs::directory_iterator dir_iter;
  for (dir_iter diriter = dir_iter(dir); diriter != dir_iter(); ++ diriter ) {
    const fs::path& path = diriter->path();
    files.push_back(path.string());
  }

  return selectCalibFile(files, runNumber, m_print_bits);

} catch (const fs::filesystem_error& ex) {
  // means cannot read directory
  return std::string();  
}

// find calibration file
std::string
CalibFileFinder::findCalibFile(const Pds::Src& src, const std::string& dataType, unsigned long runNumber) const
{
  return findCalibFile(::toString(src), dataType, runNumber);
}

// Selects calibration file from a list of file names.
std::string
CalibFileFinder::selectCalibFile(const std::vector<std::string>& files, unsigned long runNumber, unsigned print_bits)
{
  // convert strings into sortable objects
  std::vector<CalibFile> calfiles;
  for (std::vector<std::string>::const_iterator iter = files.begin(); iter != files.end(); ++ iter) {

    const fs::path path(*iter);

    // Ignore HISTORY files
    if (path.stem().string() == "HISTORY") continue;

    // only take *.data files
    if (path.extension() != ".data") {
      if( print_bits & 1 ) MsgLog(logger, info, "skipping file with wrong extension: " + path.string());
      continue;
    }

    try {
      calfiles.push_back(CalibFile(path));
    } catch (const std::exception& ex) {
      if( print_bits & 2 ) MsgLog(logger, warning, "skipping file: " + path.string() + ": " + ex.what());
    }

  }

  // find the last file in the list whose begin run is less or equal
  // to the run number
  std::sort(calfiles.begin(), calfiles.end());
  typedef std::vector<CalibFile>::const_reverse_iterator FileIter;
  for (FileIter it = calfiles.rbegin() ; it != calfiles.rend() ; ++ it ) {
    MsgLog(logger, debug, "trying: " << *it << " for run " << runNumber);
    if (it->begin() <= runNumber and runNumber <= it->end()) return it->path().string();
  }
  return std::string();
}

} // namespace PSCalib
