//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: CsPadCalibV1Cvt.cpp 1529 2011-02-16 23:59:29Z salnikov $
//
// Description:
//	Class CsPadCalibV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/CsPadCalibV1Cvt.h"

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
#include "H5DataTypes/CsPadCommonModeSubV1.h"
#include "H5DataTypes/CsPadFilterV1.h"
#include "H5DataTypes/CsPadPedestalsV1.h"
#include "H5DataTypes/CsPadPixelStatusV1.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/CalibObjectStore.h"
#include "O2OTranslator/O2OMetaData.h"
#include "pdscalibdata/CsPadCommonModeSubV1.h"
#include "pdscalibdata/CsPadFilterV1.h"
#include "pdscalibdata/CsPadPedestalsV1.h"
#include "pdscalibdata/CsPadPixelStatusV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace fs = boost::filesystem;

namespace {

  const char logger[] = "CsPadCalibV1Cvt";
  
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

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
CsPadCalibV1Cvt::CsPadCalibV1Cvt (const std::string& typeGroupName,
                                  const O2OMetaData& metadata,
                                  CalibObjectStore& calibStore)
  : DataTypeCvtI()
  , m_typeGroupName(typeGroupName)
  , m_metadata(metadata)
  , m_calibStore(calibStore)
  , m_groups()
{
}

//--------------
// Destructor --
//--------------
CsPadCalibV1Cvt::~CsPadCalibV1Cvt ()
{
}

/// main method of this class
void 
CsPadCalibV1Cvt::convert ( const void* data, 
                             size_t size,
                             const Pds::TypeId& typeId,
                             const O2OXtcSrc& src,
                             const H5DataTypes::XtcClockTime& time ) 
{
  // this should not happen
  assert ( not m_groups.empty() ) ;

  // find file with pedestals
  std::string pedFileName = findCalibFile(src, "pedestals");

  // find file with pixel status
  std::string pixFileName = findCalibFile(src, "pixel_status");
  
  // find file with common mode data
  std::string cmodeFileName = findCalibFile(src, "common_mode");
  
  // find file with filter data
  std::string filterFileName = findCalibFile(src, "filter");
  
  if ( pedFileName.empty() and pixFileName.empty() and cmodeFileName.empty()) return;

  // read everything
  boost::shared_ptr<pdscalibdata::CsPadPedestalsV1> peds;
  boost::shared_ptr<pdscalibdata::CsPadPixelStatusV1> pixStat;
  boost::shared_ptr<pdscalibdata::CsPadCommonModeSubV1> cmode;
  boost::shared_ptr<pdscalibdata::CsPadFilterV1> filter;
  if (not pedFileName.empty()) {
    peds.reset(new pdscalibdata::CsPadPedestalsV1(pedFileName));
    MsgLogRoot(info, "Read CsPad pedestals from " << pedFileName);
  }
  if (not pixFileName.empty()) {
    pixStat.reset(new pdscalibdata::CsPadPixelStatusV1(pixFileName));
    MsgLogRoot(info, "Read CsPad pixel status from " << pixFileName);
  }
  if (not cmodeFileName.empty()) {
    cmode.reset(new pdscalibdata::CsPadCommonModeSubV1(cmodeFileName));
    MsgLogRoot(info, "Read CsPad common mode data from " << cmodeFileName);
  }
  if (not filterFileName.empty()) {
    filter.reset(new pdscalibdata::CsPadFilterV1(filterFileName));
    MsgLogRoot(info, "Read CsPad filter data from " << filterFileName);
  }

  // get the name of the group for this object
  const std::string& grpName = m_typeGroupName + "/" + src.name() ;
  
  if ( m_groups.top().hasChild(m_typeGroupName) ) {
    hdf5pp::Group typeGroup = m_groups.top().openGroup(m_typeGroupName);
    if ( typeGroup.hasChild(src.name()) ) {
      MsgLog("ConfigDataTypeCvt", trace, "group " << grpName << " already exists") ;
      return;
    }
  }
  
  // create separate group
  hdf5pp::Group grp = m_groups.top().createGroup( grpName );

  // store it in a file
  if (peds.get()) {
    H5DataTypes::CsPadPedestalsV1::store(*peds, grp, pedFileName);
  }
  if (pixStat.get()) {
    H5DataTypes::CsPadPixelStatusV1::store(*pixStat, grp, pixFileName);
  }
  if (cmode.get()) {
    H5DataTypes::CsPadCommonModeSubV1::store(*cmode, grp, cmodeFileName);
  }
  if (filter.get()) {
    H5DataTypes::CsPadFilterV1::store(*filter, grp, filterFileName);
  }
  
  // store it in calibration object store
  Pds::DetInfo address = static_cast<const Pds::DetInfo&>(src.top());
  m_calibStore.add(peds, address);
  m_calibStore.add(pixStat, address);
  m_calibStore.add(cmode, address);
  m_calibStore.add(filter, address);
}

/// method called when the driver makes a new group in the file
void 
CsPadCalibV1Cvt::openGroup( hdf5pp::Group group ) 
{
  m_groups.push ( group ) ;
}

/// method called when the driver closes a group in the file
void 
CsPadCalibV1Cvt::closeGroup( hdf5pp::Group group ) 
{
  if ( m_groups.empty() ) return ;
  while ( m_groups.top() != group ) m_groups.pop() ;
  if ( m_groups.empty() ) return ;
  m_groups.pop() ;
}

// find pedestals data
std::string 
CsPadCalibV1Cvt::findCalibFile(const O2OXtcSrc& src, const std::string& datatype) const
try {
  // if no directory given then don't do anything
  if ( m_metadata.calibDir().empty() ) return std::string();
  
  // construct full path name
  fs::path dir = m_metadata.calibDir();
  dir /= m_typeGroupName;
  dir /= src.name();
  dir /= datatype;

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

  unsigned long run = m_metadata.runNumber();

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

} // namespace O2OTranslator
