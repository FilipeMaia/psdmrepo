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
#include "H5DataTypes/AcqirisConfigV1.h"
#include "H5DataTypes/CameraFrameFexConfigV1.h"
#include "H5DataTypes/CameraFrameV1.h"
#include "H5DataTypes/CameraTwoDGaussianV1.h"
#include "H5DataTypes/EvrConfigV1.h"
#include "H5DataTypes/Opal1kConfigV1.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/AcqirisDataDescV1Cvt.h"
#include "O2OTranslator/CameraFrameV1Cvt.h"
#include "O2OTranslator/ConfigDataTypeCvt.h"
#include "O2OTranslator/DataTypeCvtFactory.h"
#include "O2OTranslator/EvtDataTypeCvt.h"
#include "O2OTranslator/EvtDataTypeCvtFactory.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/O2OFileNameFactory.h"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/Level.hh"
#include "pdsdata/xtc/Sequence.hh"
#include "pdsdata/xtc/Src.hh"


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

  // create new group in a file
  template <typename Parent>
  hdf5pp::Group createGroup ( Parent parent, const std::string& groupName ) {
    MsgLog( logger, debug, "HDF5Writer -- creating group " << groupName ) ;
    hdf5pp::Group grp = parent.createGroup( groupName ) ;
    return grp ;
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {


//----------------
// Constructors --
//----------------
O2OHdf5Writer::O2OHdf5Writer ( const O2OFileNameFactory& nameFactory,
                               bool overwrite,
                               SplitMode split,
                               hsize_t splitSize,
                               bool ignoreUnknowXtc,
                               int compression )
  : O2OXtcScannerI()
  , m_nameFactory( nameFactory )
  , m_file()
  , m_state(Undefined)
  , m_mapGroup()
  , m_eventTime()
  , m_cvtMap()
  , m_ignore(ignoreUnknowXtc)
  , m_compression(compression)
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

  closeContainers() ;

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
    m_mapGroup = createGroup ( m_file, buf ) ;

    // store transition time as couple of attributes to this new group
    //::storeClock ( m_file, seq.clock() ) ;

    // switch to mapped state
    m_state = Mapped ;

  } else if ( seq.service() == Pds::TransitionId::Configure ) {

    // check current state
    if ( m_state != Mapped ) {
      throw O2OXTCTransitionException( "Configure", ::stateName(m_state) ) ;
    }

    // Create groups for data
    hdf5pp::Group configGroup = createGroup ( m_mapGroup, "Configure" ) ;
    hdf5pp::Group eventGroup = createGroup ( m_mapGroup, "EventData" ) ;


    // store transition time as couple of attributes to this new group
    //::storeClock ( m_file, seq.clock() ) ;

    // instantiate all factories for config converters
    DataTypeCvtFactoryPtr factory ;
    uint32_t typeId ;

    factory.reset( new DataTypeCvtFactory< ConfigDataTypeCvt<H5DataTypes::AcqirisConfigV1> > ( configGroup, "Acqiris::ConfigV1" ) ) ;
    typeId =  Pds::TypeId(Pds::TypeId::Id_AcqConfig,1).value() ;
    m_cvtMap.insert( CvtMap::value_type( typeId, factory ) ) ;

    factory.reset( new DataTypeCvtFactory< ConfigDataTypeCvt<H5DataTypes::Opal1kConfigV1> > ( configGroup, "Opal1k::ConfigV1" ) ) ;
    typeId =  Pds::TypeId(Pds::TypeId::Id_Opal1kConfig,1).value() ;
    m_cvtMap.insert( CvtMap::value_type( typeId, factory ) ) ;

    factory.reset( new DataTypeCvtFactory< ConfigDataTypeCvt<H5DataTypes::CameraFrameFexConfigV1> > ( configGroup, "Camera::FrameFexConfigV1" ) ) ;
    typeId =  Pds::TypeId(Pds::TypeId::Id_FrameFexConfig,1).value() ;
    m_cvtMap.insert( CvtMap::value_type( typeId, factory ) ) ;

    factory.reset( new DataTypeCvtFactory< ConfigDataTypeCvt<H5DataTypes::EvrConfigV1> > ( configGroup, "EvrData::ConfigV1" ) ) ;
    typeId =  Pds::TypeId(Pds::TypeId::Id_EvrConfig,1).value() ;
    m_cvtMap.insert( CvtMap::value_type( typeId, factory ) ) ;

    // very special converter for Acqiris::DataDescV1, it needs two types of data
    hsize_t chunk_size = 128*1024 ;
    factory.reset( new EvtDataTypeCvtFactory< AcqirisDataDescV1Cvt > (
        eventGroup, "Acqiris::DataDescV1", chunk_size, m_compression ) ) ;
    typeId =  Pds::TypeId(Pds::TypeId::Id_AcqConfig,1).value() ;
    m_cvtMap.insert( CvtMap::value_type( typeId, factory ) ) ;
    typeId =  Pds::TypeId(Pds::TypeId::Id_AcqWaveform,1).value() ;
    m_cvtMap.insert( CvtMap::value_type( typeId, factory ) ) ;

    // switch to configured state
    m_state = Configured ;

  } else if ( seq.service() == Pds::TransitionId::BeginRun ) {

    // check current state
    if ( m_state != Configured ) {
      throw O2OXTCTransitionException( "BeginRun", ::stateName(m_state) ) ;
    }

    // switch to running state
    m_state = Running ;

    // instantiate all factories for event converters
    DataTypeCvtFactoryPtr factory ;
    uint32_t typeId ;
    hsize_t chunk_size = 128*1024 ;

    // event group should have been created already
    hdf5pp::Group eventGroup = m_mapGroup.openGroup( "EventData" ) ;

    factory.reset( new EvtDataTypeCvtFactory< EvtDataTypeCvt<H5DataTypes::CameraTwoDGaussianV1> > (
        eventGroup, "Camera::TwoDGaussianV1", chunk_size, m_compression ) ) ;
    typeId =  Pds::TypeId(Pds::TypeId::Id_TwoDGaussian,1).value() ;
    m_cvtMap.insert( CvtMap::value_type( typeId, factory ) ) ;

    factory.reset( new EvtDataTypeCvtFactory< CameraFrameV1Cvt > (
        eventGroup, "Camera::FrameV1", chunk_size, m_compression ) ) ;
    typeId =  Pds::TypeId(Pds::TypeId::Id_Frame,1).value() ;
    m_cvtMap.insert( CvtMap::value_type( typeId, factory ) ) ;

  } else if ( seq.service() == Pds::TransitionId::L1Accept ) {

    // store current event time
    m_eventTime = H5DataTypes::XtcClockTime(seq.clock()) ;

  } else if ( seq.service() == Pds::TransitionId::EndRun ) {

    // switch back to configured state
    m_state = Configured ;

  } else if ( seq.service() == Pds::TransitionId::Unconfigure ) {

    // switch back to mapped state
    m_state = Mapped ;

  } else if ( seq.service() == Pds::TransitionId::Unmap ) {

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
O2OHdf5Writer::dataObject ( const void* data, const Pds::TypeId& typeId, const Pds::DetInfo& detInfo )
{
  // find this type in the converter factory map
  CvtMap::iterator it = m_cvtMap.find( typeId.value() ) ;
  if ( it != m_cvtMap.end() ) {

    do {

      DataTypeCvtFactoryPtr factory = it->second ;
      DataTypeCvtI* converter = factory->converter( detInfo ) ;
      converter->convert( data, typeId, detInfo, m_eventTime ) ;

      ++ it ;

    } while ( it != m_cvtMap.end() and it->first == typeId.value() ) ;

  } else {

    if ( m_ignore ) {
      MsgLogRoot( warning, "O2OXtcIterator::process -- unexpected type or version: "
                  << Pds::TypeId::name(typeId.id()) << "/" << typeId.version() ) ;
    } else {
      MsgLogRoot( error, "O2OXtcIterator::process -- unexpected type or version: "
                  << Pds::TypeId::name(typeId.id()) << "/" << typeId.version() ) ;
    }

  }

}

// close all containers
void
O2OHdf5Writer::closeContainers()
{
  m_cvtMap.clear() ;
}

} // namespace O2OTranslator
