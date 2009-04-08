//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OHdf5Writer...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OHdf5Writer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "H5DataTypes/AcqirisConfigV1.h"
#include "H5DataTypes/CameraFrameFexConfigV1.h"
#include "H5DataTypes/CameraTwoDGaussianV1.h"
#include "H5DataTypes/EvrConfigV1.h"
#include "H5DataTypes/Opal1kConfigV1.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/O2OFileNameFactory.h"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/Level.hh"
#include "pdsdata/xtc/Sequence.hh"
#include "pdsdata/xtc/Src.hh"
#include "pdsdata/acqiris/ConfigV1.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace O2OTranslator ;

namespace {

  const char* logger = "HDF5Writer" ;

  // printable state name
  const char* stateName ( O2OHdf5Writer::State state ) {
    switch ( state ) {
      case O2OHdf5Writer::Undefined :
        return "Undefined" ;
      case O2OHdf5Writer::Mapped :
        return "Mapped" ;
      case O2OHdf5Writer::Configured :
        return "Configured" ;
      case O2OHdf5Writer::Running :
        return "Running" ;
    }
    return "*ERROR*" ;
  }

  // Class name for new groups depends on the state
  const char* className ( O2OHdf5Writer::State state ) {
    switch ( state ) {
      case O2OHdf5Writer::Undefined :
        return "NXentry" ;
      case O2OHdf5Writer::Mapped :
        return "NXinstrument" ;
      case O2OHdf5Writer::Configured :
        return "CfgObject" ;
      case O2OHdf5Writer::Running :
        return "NXevent_data" ;
    }
    return "Unexpected" ;
  }

  // get the group name from the source
  std::string groupName( const char* typeName, const Pds::DetInfo& info ) {
    std::ostringstream grp ;
    grp << typeName << '/' << Pds::DetInfo::name(info.detector()) << '.' << info.detId()
        << ':' << Pds::DetInfo::name(info.device()) << '.' << info.devId() ;

    return grp.str() ;
  }

  // create new group in a file
  template <typename Parent>
  hdf5pp::Group createGroup ( Parent parent, const std::string& groupName, O2OHdf5Writer::State state ) {
    const char* gclass = className(state) ;
    MsgLog( logger, debug, "HDF5Writer -- creating group " << groupName << "/" << gclass ) ;
    hdf5pp::Group grp = parent.createGroup( groupName ) ;

    // make an attribute
    hdf5pp::Attribute<const char*> clsAttr = grp.createAttr<const char*> ( "NX_class" ) ;
    clsAttr.store ( gclass ) ;

    return grp ;
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

// comparator for DetInfo objects
bool
O2OHdf5Writer::CmpDetInfo::operator()( const Pds::DetInfo& lhs, const Pds::DetInfo& rhs ) const
{
  int ldet = lhs.detector();
  int rdet = rhs.detector();
  if ( ldet < rdet ) return true ;
  if ( ldet > rdet ) return false ;

  int ldev = lhs.device();
  int rdev = rhs.device();
  if ( ldev < rdev ) return true ;
  if ( ldev > rdev ) return false ;

  uint32_t ldetId = lhs.detId();
  uint32_t rdetId = rhs.detId();
  if ( ldetId < rdetId ) return true ;
  if ( ldetId > rdetId ) return false ;

  uint32_t ldevId = lhs.devId();
  uint32_t rdevId = rhs.devId();
  if ( ldevId < rdevId ) return true ;
  if ( ldevId > rdevId ) return false ;

  uint32_t lpid = lhs.processId();
  uint32_t rpid = rhs.processId();
  if ( lpid < rpid ) return true ;
  return false ;
}


//----------------
// Constructors --
//----------------
O2OHdf5Writer::O2OHdf5Writer ( const O2OFileNameFactory& nameFactory,
                               bool overwrite,
                               SplitMode split,
                               hsize_t splitSize )
  : O2OXtcScannerI()
  , m_nameFactory( nameFactory )
  , m_file()
  , m_state(Undefined)
  , m_mapGroup()
  , m_configGroup()
  , m_eventTime()
  , m_cameraTwoDGaussianV1Cont()
  , m_cameraTwoDGaussianV1TimeCont()
{
  std::string fileTempl = m_nameFactory.makeH5Path () ;
  MsgLog( logger, debug, "O2OHdf5Writer - open output file " << fileTempl ) ;

  // Disable printing of error messages
  //stat = H5Eset_auto2( H5E_DEFAULT, 0, 0 ) ;

  // we want to create new file
  hdf5pp::PListFileAccess fapl ;
  if ( split == Family ) {
    // use FAMILY driver
    fapl.set_family_driver ( splitSize, hdf5pp::PListFileAccess() ) ;
  }

  hdf5pp::File::CreateMode mode = overwrite ? hdf5pp::File::Truncate : hdf5pp::File::Exclusive ;
  m_file = hdf5pp::File::create ( fileTempl, mode, hdf5pp::PListFileCreate(), fapl ) ;

}

//--------------
// Destructor --
//--------------
O2OHdf5Writer::~O2OHdf5Writer ()
{
  MsgLog( logger, debug, "O2OHdf5Writer - close output file" ) ;

  m_cameraTwoDGaussianV1Cont.reset() ;
  m_cameraTwoDGaussianV1TimeCont.reset() ;

  m_file.close() ;

}

// signal start/end of the event (datagram)
void
O2OHdf5Writer::eventStart ( const Pds::Sequence& seq )
{
  MsgLog( logger, debug, "O2OHdf5Writer::eventStart " << Pds::TransitionId::name(seq.service())
          << " seq.type=" << seq.type()
          << " seq.service=" << Pds::TransitionId::name(seq.service()) ) ;

  if ( seq.service() == Pds::TransitionId::Map ) {

    // check current state
    if ( m_state != Undefined ) {
      throw O2OXTCTransitionException( "Map", ::stateName(m_state) ) ;
    }

    // dump seconds as a hex string, it will be group name
    char buf[32] ;
    snprintf ( buf, sizeof buf, "Map:%08X", seq.clock().seconds() ) ;

    // create NXentry group
    m_mapGroup = createGroup ( m_file, buf, m_state ) ;

    // store transition time as couple of attributes to this new group
    //::storeClock ( m_file, seq.clock() ) ;

    // switch to mapped state
    m_state = Mapped ;

  } else if ( seq.service() == Pds::TransitionId::Configure ) {

    // check current state
    if ( m_state != Mapped ) {
      throw O2OXTCTransitionException( "Configure", ::stateName(m_state) ) ;
    }

    // Create 'NXinstrument' group
    m_configGroup = createGroup ( m_mapGroup, "Configure", m_state ) ;

    // store transition time as couple of attributes to this new group
    //::storeClock ( m_file, seq.clock() ) ;

    // switch to configured state
    m_state = Configured ;

  } else if ( seq.service() == Pds::TransitionId::BeginRun ) {

    // check current state
    if ( m_state != Configured ) {
      throw O2OXTCTransitionException( "BeginRun", ::stateName(m_state) ) ;
    }

    // switch to running state
    m_state = Running ;

  } else if ( seq.service() == Pds::TransitionId::L1Accept ) {

    m_eventTime = H5DataTypes::XtcClockTime(seq.clock()) ;

  } else if ( seq.service() == Pds::TransitionId::EndRun ) {

    // switch back to configured state
    m_state = Configured ;

  } else if ( seq.service() == Pds::TransitionId::Unconfigure ) {

    // switch back to mapped state
    m_state = Mapped ;

  } else if ( seq.service() == Pds::TransitionId::Unmap ) {

    // close all storers
    m_cameraTwoDGaussianV1Cont.reset() ;
    m_cameraTwoDGaussianV1TimeCont.reset() ;

    // close 'NXentry' group, go back to top level
    m_mapGroup.close() ;

    // switch back to undetermined state
    m_state = Undefined ;

  }

  MsgLog( logger, debug, "O2OHdf5Writer -- now in the state " << ::stateName(m_state) ) ;
}

void
O2OHdf5Writer::eventEnd ( const Pds::Sequence& seq )
{
  MsgLog( logger, debug, "O2OHdf5Writer::eventEnd " << Pds::TransitionId::name(seq.service()) ) ;

  if ( seq.service() == Pds::TransitionId::Configure ) {

    // check current state
    if ( m_state != Configured ) {
      throw O2OXTCTransitionException( "Configure", ::stateName(m_state) ) ;
    }

    // close 'Configure' group
    m_configGroup.close() ;
  }
}

// signal start/end of the level
void
O2OHdf5Writer::levelStart ( const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::levelStart " << Pds::Level::name(src.level()) ) ;
}

void
O2OHdf5Writer::levelEnd ( const Pds::Src& src )
{
  MsgLog( logger, debug, "O2OHdf5Writer::levelEnd " << Pds::Level::name(src.level()) ) ;
}

// visit the data object
void
O2OHdf5Writer::dataObject ( const Pds::Acqiris::ConfigV1& data, const Pds::DetInfo& detInfo )
{
  // get the group name
  const std::string& grpName = ::groupName( "Acqiris::ConfigV1", detInfo ) ;

  MsgLog( logger, debug, "O2OHdf5Writer::dataObject " << grpName ) ;

  // define separate group
  hdf5pp::Group grp = m_configGroup.createGroup( grpName );
  grp.createAttr<const char*> ( "class" ).store ( "Acqiris::ConfigV1" ) ;

  // store the data
  H5DataTypes::storeAcqirisConfigV1 ( data, grp ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::Acqiris::DataDescV1& data, const Pds::DetInfo& detInfo )
{
  // get the group name
  const std::string& grpName = ::groupName( "Acqiris::DataDescV1", detInfo ) ;

  MsgLog( logger, debug, "O2OHdf5Writer::dataObject " << grpName ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::Camera::FrameFexConfigV1& data, const Pds::DetInfo& detInfo )
{
  // get the group name
  const std::string& grpName = ::groupName( "Camera::FrameFexConfigV1", detInfo ) ;

  MsgLog( logger, debug, "O2OHdf5Writer::dataObject " << grpName ) ;

  // define separate group
  hdf5pp::Group grp = m_configGroup.createGroup( grpName );
  grp.createAttr<const char*> ( "class" ).store ( "Camera::FrameFexConfigV1" ) ;

  // store the data
  H5DataTypes::storeCameraFrameFexConfigV1 ( data, grp ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::Camera::FrameV1& data, const Pds::DetInfo& detInfo )
{
  // get the group name
  const std::string& grpName = ::groupName( "Camera::FrameV1", detInfo ) ;

  MsgLog( logger, debug, "O2OHdf5Writer::dataObject " << grpName ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::Camera::TwoDGaussianV1& data, const Pds::DetInfo& detInfo )
{
  // get the group name
  const std::string& grpName = ::groupName( "Camera::TwoDGaussianV1", detInfo ) ;

  MsgLog( logger, debug, "O2OHdf5Writer::dataObject " << grpName ) ;

  if ( not m_cameraTwoDGaussianV1Cont.get() ) {

    // define two containers in a separate group
    hdf5pp::Group contGrp = m_mapGroup.createGroup(grpName);
    contGrp.createAttr<const char*> ( "class" ).store ( "Camera::TwoDGaussianV1" ) ;

    // chunk size will be 1M/size of object
    hsize_t chunk_size = 1024*1024 / sizeof(H5DataTypes::CameraTwoDGaussianV1) ;
    int deflate = 6 ;

    // make container for data objects
    hdf5pp::Type type = H5DataTypes::CameraTwoDGaussianV1::persType() ;
    m_cameraTwoDGaussianV1Cont.reset( new CameraTwoDGaussianV1Cont ( "data", contGrp, type, chunk_size, deflate ) ) ;

    // make container for time
    hdf5pp::Type timeType = H5DataTypes::XtcClockTime::persType() ;
    m_cameraTwoDGaussianV1TimeCont.reset( new XtcClockTimeCont ( "time", contGrp, timeType, chunk_size, deflate ) ) ;

  }

  // store the data in the containers
  m_cameraTwoDGaussianV1Cont->append ( H5DataTypes::CameraTwoDGaussianV1(data) ) ;
  m_cameraTwoDGaussianV1TimeCont->append ( m_eventTime ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::EvrData::ConfigV1& data, const Pds::DetInfo& detInfo )
{
  // get the group name
  const std::string& grpName = ::groupName( "EvrData::ConfigV1", detInfo ) ;

  MsgLog( logger, debug, "O2OHdf5Writer::dataObject " << grpName ) ;

  // define separate group
  hdf5pp::Group grp = m_configGroup.createGroup( grpName );
  grp.createAttr<const char*> ( "class" ).store ( "EvrData::ConfigV1" ) ;

  // store the data
  H5DataTypes::storeEvrConfigV1( data, grp ) ;
}

void
O2OHdf5Writer::dataObject ( const Pds::Opal1k::ConfigV1& data, const Pds::DetInfo& detInfo )
{
  // get the group name
  const std::string& grpName = ::groupName( "Opal1k::ConfigV1", detInfo ) ;

  MsgLog( logger, debug, "O2OHdf5Writer::dataObject " << grpName ) ;

  // define separate group
  hdf5pp::Group grp = m_configGroup.createGroup( grpName );
  grp.createAttr<const char*> ( "class" ).store ( "Opal1k::ConfigV1" ) ;

  // store the data
  H5DataTypes::storeOpal1kConfigV1( data, grp ) ;
}

} // namespace O2OTranslator
