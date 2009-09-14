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
#include <uuid/uuid.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/AcqirisConfigV1.h"
#include "H5DataTypes/CameraFrameFexConfigV1.h"
#include "H5DataTypes/CameraFrameV1.h"
#include "H5DataTypes/CameraTwoDGaussianV1.h"
#include "H5DataTypes/EvrConfigV1.h"
#include "H5DataTypes/Opal1kConfigV1.h"
#include "H5DataTypes/PulnixTM6740ConfigV1.h"
#include "LusiTime/Time.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/AcqirisDataDescV1Cvt.h"
#include "O2OTranslator/CameraFrameV1Cvt.h"
#include "O2OTranslator/ConfigDataTypeCvt.h"
#include "O2OTranslator/EvtDataTypeCvtDef.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/O2OFileNameFactory.h"
#include "O2OTranslator/O2OMetaData.h"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/Level.hh"
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
      case O2OHdf5Writer::CalibCycle :
        return "CalibCycle" ;
    }
    return "*ERROR*" ;
  }

  // store time as attributes to the group
  void storeClock ( hdf5pp::Group group, const Pds::ClockTime& clock, const std::string& what )
  {
    hdf5pp::Attribute<uint32_t> attr1 = group.createAttr<uint32_t> ( what+".seconds" ) ;
    attr1.store ( clock.seconds() ) ;
    hdf5pp::Attribute<uint32_t> attr2 = group.createAttr<uint32_t> ( what+".nanoseconds" ) ;
    attr2.store ( clock.nanoseconds() ) ;
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
                               int compression,
                               bool extGroups,
                               const O2OMetaData& metadata )
  : O2OXtcScannerI()
  , m_nameFactory( nameFactory )
  , m_file()
  , m_state()
  , m_groups()
  , m_eventTime()
  , m_cvtMap()
  , m_compression(compression)
  , m_extGroups(extGroups)
  , m_metadata(metadata)
{
  std::string fileTempl = m_nameFactory.makeH5Path ( split != NoSplit ) ;
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

  // add UUID to the file attributes
  uuid_t uuid ;
  uuid_generate( uuid );
  char uuid_buf[64] ;
  uuid_unparse ( uuid, uuid_buf ) ;
  m_file.createAttr<const char*> ("UUID").store ( uuid_buf ) ;

  // add some metadata to the top group
  LusiTime::Time ctime = LusiTime::Time::now() ;
  m_file.createAttr<const char*> ("origin").store ( "translator" ) ;
  m_file.createAttr<const char*> ("created").store ( ctime.toString().c_str() ) ;

  m_file.createAttr<uint32_t> ("runNumber").store ( m_metadata.runNumber() ) ;
  m_file.createAttr<const char*> ("runType").store ( m_metadata.runType().c_str() ) ;
  m_file.createAttr<const char*> ("experiment").store ( m_metadata.experiment().c_str() ) ;

  // we are in bad state, this state should never be popped
  m_state.push(Undefined) ;

  // store top group
  m_groups.push ( m_file.openGroup("/") ) ;

  typedef O2OMetaData::const_iterator MDIter ;
  for ( MDIter it = m_metadata.extra_begin() ; it != m_metadata.extra_end() ; ++ it ) {
    try {
      m_file.createAttr<const char*> (it->first).store ( it->second.c_str() ) ;
    } catch ( std::exception& e ) {
      // this is not fatal, just print error message and continue
      MsgLog( logger, error, "failed to store metadata: " << e.what()
          << "\n\tkey='" << it->first << "', value='" << it->second << "'" ) ;
    }
  }

  // instantiate all factories
  DataTypeCvtPtr converter ;
  uint32_t typeId ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::AcqirisConfigV1> ( "Acqiris::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_AcqConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::Opal1kConfigV1> ( "Opal1k::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_Opal1kConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::PulnixTM6740ConfigV1> ( "Pulnix::TM6740ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_TM6740Config,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::CameraFrameFexConfigV1> ( "Camera::FrameFexConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_FrameFexConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  converter.reset( new ConfigDataTypeCvt<H5DataTypes::EvrConfigV1> ( "EvrData::ConfigV1" ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_EvrConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  hsize_t chunk_size = 128*1024 ;

  // instantiate all factories for event converters
  converter.reset( new EvtDataTypeCvtDef<H5DataTypes::CameraTwoDGaussianV1> (
      "Camera::TwoDGaussianV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_TwoDGaussian,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // special converter for CameraFrame type
  converter.reset( new CameraFrameV1Cvt ( "Camera::FrameV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_Frame,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

  // very special converter for Acqiris::DataDescV1, it needs two types of data
  converter.reset( new AcqirisDataDescV1Cvt ( "Acqiris::DataDescV1", chunk_size, m_compression ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_AcqConfig,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;
  typeId =  Pds::TypeId(Pds::TypeId::Id_AcqWaveform,1).value() ;
  m_cvtMap.insert( CvtMap::value_type( typeId, converter ) ) ;

}

//--------------
// Destructor --
//--------------
O2OHdf5Writer::~O2OHdf5Writer ()
{
  MsgLog( logger, debug, "O2OHdf5Writer - close output file" ) ;

  m_cvtMap.clear() ;
  m_file.close() ;
}

// signal start/end of the event (datagram)
void
O2OHdf5Writer::eventStart ( const Pds::Dgram& dgram )
{
  MsgLog( logger, debug, "O2OHdf5Writer::eventStart " << Pds::TransitionId::name(dgram.seq.service())
          << " dgram.seq.type=" << dgram.seq.type()
          << " dgram.seq.service=" << Pds::TransitionId::name(dgram.seq.service()) ) ;

  switch ( dgram.seq.service()  ) {

    case Pds::TransitionId::Map :

      // close all states
      this->closeGroup( dgram, CalibCycle ) ;
      this->closeGroup( dgram, Running ) ;
      this->closeGroup( dgram, Configured ) ;
      this->closeGroup( dgram, Mapped ) ;
      this->openGroup( dgram, Mapped ) ;

      break ;

    case Pds::TransitionId::Unmap :

      // close all states
      this->closeGroup( dgram, CalibCycle ) ;
      this->closeGroup( dgram, Running ) ;
      this->closeGroup( dgram, Configured ) ;
      this->closeGroup( dgram, Mapped ) ;

      break ;

    case Pds::TransitionId::Configure :

      // close all states up to Mapped
      this->closeGroup( dgram, CalibCycle ) ;
      this->closeGroup( dgram, Running ) ;
      this->closeGroup( dgram, Configured ) ;
      this->openGroup( dgram, Configured ) ;

      break ;

    case Pds::TransitionId::Unconfigure :

      // close all states up to Mapped
      this->closeGroup( dgram, CalibCycle ) ;
      this->closeGroup( dgram, Running ) ;
      this->closeGroup( dgram, Configured ) ;
      this->closeGroup( dgram, Mapped ) ;

      break ;

    case Pds::TransitionId::BeginRun :

      // close all states up to Configured
      this->closeGroup( dgram, CalibCycle ) ;
      this->closeGroup( dgram, Running ) ;
      this->openGroup( dgram, Running ) ;

      break ;

    case Pds::TransitionId::EndRun :

      // close all states up to Configured
      this->closeGroup( dgram, CalibCycle ) ;
      this->closeGroup( dgram, Running ) ;

      break ;

    case Pds::TransitionId::BeginCalibCycle :

      // close all states up to Running
      this->closeGroup( dgram, CalibCycle ) ;
      this->openGroup( dgram, CalibCycle ) ;

      break ;

    case Pds::TransitionId::EndCalibCycle :

      // close all states up to Running
      this->closeGroup( dgram, CalibCycle ) ;

      break ;

    case Pds::TransitionId::L1Accept :

      // store current event time
      m_eventTime = H5DataTypes::XtcClockTime(dgram.seq.clock()) ;

      break ;

    case Pds::TransitionId::Enable :
    case Pds::TransitionId::Disable :
    case Pds::TransitionId::Unknown :
    case Pds::TransitionId::Reset :
    case Pds::TransitionId::NumberOf :

      break ;
  }

  MsgLog( logger, debug, "O2OHdf5Writer -- now in the state " << ::stateName(m_state.top()) ) ;
}

void
O2OHdf5Writer::eventEnd ( const Pds::Dgram& dgram )
{
}


void
O2OHdf5Writer::openGroup ( const Pds::Dgram& dgram, State state )
{
  // create group
  const std::string& name = groupName ( state, dgram.seq.clock() ) ;
  MsgLog( logger, debug, "HDF5Writer -- creating group " << name ) ;
  hdf5pp::Group group = m_groups.top().createGroup( name ) ;

  // store transition time as couple of attributes to this new group
  ::storeClock ( group, dgram.seq.clock(), "start" ) ;

  // switch to mapped state
  m_state.push(state) ;
  m_groups.push( group ) ;

  // notify all converters
  for ( CvtMap::iterator it = m_cvtMap.begin() ; it != m_cvtMap.end() ; ++ it ) {
    it->second->openGroup( group ) ;
  }
}

void
O2OHdf5Writer::closeGroup ( const Pds::Dgram& dgram, State state )
{
  if ( m_state.top() != state ) return ;

  // store transition time as couple of attributes to this new group
  ::storeClock ( m_groups.top(), dgram.seq.clock(), "end" ) ;

  // notify all converters
  for ( CvtMap::iterator it = m_cvtMap.begin() ; it != m_cvtMap.end() ; ++ it ) {
    it->second->closeGroup( m_groups.top() ) ;
  }

  // close the group
  m_groups.top().close() ;

  // switch back to previous state
  m_state.pop() ;
  m_groups.pop() ;
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
  // find this type in the converter map
  CvtMap::iterator it = m_cvtMap.find( typeId.value() ) ;
  if ( it != m_cvtMap.end() ) {

    do {

      DataTypeCvtPtr converter = it->second ;
      converter->convert( data, typeId, detInfo, m_eventTime ) ;

      ++ it ;

    } while ( it != m_cvtMap.end() and it->first == typeId.value() ) ;

  } else {

    MsgLogRoot( error, "O2OXtcIterator::process -- unexpected type or version: "
                << Pds::TypeId::name(typeId.id()) << "/" << typeId.version() ) ;

  }

}

// Construct a group name
std::string
O2OHdf5Writer::groupName( State state, const Pds::ClockTime& clock ) const
{
  const char* prefix = "Undefined" ;
  switch ( state ) {
    case O2OHdf5Writer::Mapped :
      prefix = "Map" ;
      break ;
    case O2OHdf5Writer::Configured :
      prefix = "Configure" ;
      break ;
    case O2OHdf5Writer::Running :
      prefix = "Run" ;
      break ;
    case O2OHdf5Writer::CalibCycle :
      prefix = "CalibCycle" ;
      break ;
    case O2OHdf5Writer::Undefined :
    default :
      prefix = "Undefined" ;
      break ;
  }

  if ( m_extGroups ) {
    // dump seconds as a hex string, it will be group name
    char buf[32] ;
    snprintf ( buf, sizeof buf, "%s:%08X", prefix, clock.seconds() ) ;
    return buf;
  } else {
    return prefix;
  }
}

} // namespace O2OTranslator
