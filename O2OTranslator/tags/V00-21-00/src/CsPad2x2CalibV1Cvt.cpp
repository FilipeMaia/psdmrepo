//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2CalibV1Cvt...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/CsPad2x2CalibV1Cvt.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/CsPadCommonModeSubV1.h"
#include "H5DataTypes/CsPadFilterV1.h"
#include "H5DataTypes/CsPad2x2PedestalsV1.h"
#include "H5DataTypes/CsPad2x2PixelStatusV1.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/CalibObjectStore.h"
#include "O2OTranslator/O2OMetaData.h"
#include "pdscalibdata/CsPadCommonModeSubV1.h"
#include "pdscalibdata/CsPadFilterV1.h"
#include "pdscalibdata/CsPad2x2PedestalsV1.h"
#include "pdscalibdata/CsPad2x2PixelStatusV1.h"
#include "PSCalib/CalibFileFinder.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "CsPad2x2CalibV1Cvt";
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
CsPad2x2CalibV1Cvt::CsPad2x2CalibV1Cvt (const std::string& typeGroupName,
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
CsPad2x2CalibV1Cvt::~CsPad2x2CalibV1Cvt ()
{
}

/// main method of this class
void 
CsPad2x2CalibV1Cvt::convert ( const void* data,
                               size_t size,
                               const Pds::TypeId& typeId,
                               const O2OXtcSrc& src,
                               const H5DataTypes::XtcClockTimeStamp& time )
{
  // this should not happen
  assert ( not m_groups.empty() ) ;

  {
    // We only accept data from Cspad2x2 devices here
    const Pds::Src& pdssrc = src.top();
    if (pdssrc.level() != Pds::Level::Source) {
      MsgLog(logger, warning, "CsPad2x2CalibV1Cvt::convert called with non-Source level: " << Pds::Level::name(pdssrc.level()));
      return;
    }
    const Pds::DetInfo& dinfo = static_cast<const Pds::DetInfo&>(pdssrc);
    if (dinfo.device() != Pds::DetInfo::Cspad2x2) return;
  }
  
  PSCalib::CalibFileFinder calibFileFinder(m_metadata.calibDir(), m_typeGroupName);

  // find file with pedestals
  std::string pedFileName = calibFileFinder.findCalibFile(src.name(), "pedestals", m_metadata.runNumber());

  // find file with pixel status
  std::string pixFileName = calibFileFinder.findCalibFile(src.name(), "pixel_status", m_metadata.runNumber());
  
  // find file with common mode data
  std::string cmodeFileName = calibFileFinder.findCalibFile(src.name(), "common_mode", m_metadata.runNumber());
  
  // find file with filter data
  std::string filterFileName = calibFileFinder.findCalibFile(src.name(), "filter", m_metadata.runNumber());
  
  if ( pedFileName.empty() and pixFileName.empty() and cmodeFileName.empty()) return;

  // read everything
  boost::shared_ptr<pdscalibdata::CsPad2x2PedestalsV1> peds;
  boost::shared_ptr<pdscalibdata::CsPad2x2PixelStatusV1> pixStat;
  boost::shared_ptr<pdscalibdata::CsPadCommonModeSubV1> cmode;
  boost::shared_ptr<pdscalibdata::CsPadFilterV1> filter;
  if (not pedFileName.empty()) {
    peds.reset(new pdscalibdata::CsPad2x2PedestalsV1(pedFileName));
    MsgLog(logger, info, "Read CsPad pedestals from " << pedFileName);
  }
  if (not pixFileName.empty()) {
    pixStat.reset(new pdscalibdata::CsPad2x2PixelStatusV1(pixFileName));
    MsgLog(logger, info, "Read CsPad pixel status from " << pixFileName);
  }
  if (not cmodeFileName.empty()) {
    cmode.reset(new pdscalibdata::CsPadCommonModeSubV1(cmodeFileName));
    MsgLog(logger, info, "Read CsPad common mode data from " << cmodeFileName);
  }
  if (not filterFileName.empty()) {
    filter.reset(new pdscalibdata::CsPadFilterV1(filterFileName));
    MsgLog(logger, info, "Read CsPad filter data from " << filterFileName);
  }

  // get the name of the group for this object
  const std::string& grpName = m_typeGroupName + "/" + src.name() ;
  
  if ( m_groups.top().hasChild(m_typeGroupName) ) {
    hdf5pp::Group typeGroup = m_groups.top().openGroup(m_typeGroupName);
    if ( typeGroup.hasChild(src.name()) ) {
      MsgLog(logger, trace, "group " << grpName << " already exists") ;
      return;
    }
  }
  
  // create separate group
  hdf5pp::Group grp = m_groups.top().createGroup( grpName );

  // store it in a file
  if (peds.get()) {
    H5DataTypes::CsPad2x2PedestalsV1::store(*peds, grp, pedFileName);
  }
  if (pixStat.get()) {
    H5DataTypes::CsPad2x2PixelStatusV1::store(*pixStat, grp, pixFileName);
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
CsPad2x2CalibV1Cvt::openGroup( hdf5pp::Group group )
{
  m_groups.push ( group ) ;
}

/// method called when the driver closes a group in the file
void 
CsPad2x2CalibV1Cvt::closeGroup( hdf5pp::Group group )
{
  if ( m_groups.empty() ) return ;
  while ( m_groups.top() != group ) m_groups.pop() ;
  if ( m_groups.empty() ) return ;
  m_groups.pop() ;
}

} // namespace O2OTranslator
