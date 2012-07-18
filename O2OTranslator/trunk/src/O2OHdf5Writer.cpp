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

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OHdf5Writer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <uuid/uuid.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "LusiTime/Time.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OCvtFactory.h"
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
namespace fs = boost::filesystem;

namespace {

  const char* logger = "HDF5Writer" ;

  // printable state name
  const char* stateName ( O2OHdf5Writer::State state ) {
    switch ( state ) {
      case O2OHdf5Writer::Undefined :
        return "Undefined" ;
      case O2OHdf5Writer::Configured :
        return "Configured" ;
      case O2OHdf5Writer::Running :
        return "Running" ;
      case O2OHdf5Writer::CalibCycle :
        return "CalibCycle" ;
      case O2OHdf5Writer::NumberOfStates :
        break ;
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
                               const O2OMetaData& metadata,
                               const std::string& finalDir )
  : O2OXtcScannerI()
  , m_nameFactory( nameFactory )
  , m_overwrite(overwrite)
  , m_split(split)
  , m_splitSize(splitSize)
  , m_compression(compression)
  , m_extGroups(extGroups)
  , m_metadata(metadata)
  , m_finalDir(finalDir)
  , m_file()
  , m_state()
  , m_groups()
  , m_eventTime()
  , m_cvtMap()
  , m_stateCounters()
  , m_transition(Pds::TransitionId::Unknown)
  , m_configStore()
  , m_calibStore()
  , m_transClock()
{
  std::fill_n(m_stateCounters, int(NumberOfStates), 0U);
  std::fill_n(m_transClock, int(Pds::TransitionId::NumberOf), LusiTime::Time(0,0));
    
  // we are in bad state, this state should never be popped
  m_state.push(Undefined) ;

  // instantiate all converters
  O2OCvtFactory::makeConverters(m_cvtMap, m_configStore, m_calibStore, m_metadata, m_compression);

  // open new file
  openFile();
}

//--------------
// Destructor --
//--------------
O2OHdf5Writer::~O2OHdf5Writer ()
{
  MsgLog( logger, debug, "O2OHdf5Writer - close output file" ) ;

  m_cvtMap.clear() ;
  closeFile();
}

// signal start/end of the event (datagram)
bool
O2OHdf5Writer::eventStart ( const Pds::Dgram& dgram )
{
  MsgLog( logger, debug, "O2OHdf5Writer::eventStart " << Pds::TransitionId::name(dgram.seq.service())
          << " dgram.seq.type=" << dgram.seq.type()
          << " dgram.seq.service=" << Pds::TransitionId::name(dgram.seq.service()) ) ;

  m_transition = dgram.seq.service();
  Pds::ClockTime clock = dgram.seq.clock();
  LusiTime::Time t(clock.seconds(), clock.nanoseconds());
  
  // store current event time
  m_eventTime = H5DataTypes::XtcClockTime(clock) ;
  

  bool skip = false;
  switch ( m_transition ) {

    case Pds::TransitionId::Configure :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Undefined
        this->closeGroup( dgram, CalibCycle ) ;
        this->closeGroup( dgram, Running ) ;
        this->closeGroup( dgram, Configured ) ;
        this->openGroup( dgram, Configured ) ;
      }
      break ;

    case Pds::TransitionId::Unconfigure :

      if ( t != m_transClock[m_transition] ) {
         // close all states up to Mapped
        this->closeGroup( dgram, CalibCycle ) ;
        this->closeGroup( dgram, Running ) ;
        this->closeGroup( dgram, Configured ) ;
      }
      break ;

    case Pds::TransitionId::BeginRun :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Configured
        this->closeGroup( dgram, CalibCycle ) ;
        this->closeGroup( dgram, Running ) ;
        this->openGroup( dgram, Running ) ;
      }
      break ;

    case Pds::TransitionId::EndRun :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Configured
        this->closeGroup( dgram, CalibCycle ) ;
        this->closeGroup( dgram, Running ) ;
      }
      break ;

    case Pds::TransitionId::BeginCalibCycle :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Running
        this->closeGroup( dgram, CalibCycle ) ;
        this->openGroup( dgram, CalibCycle ) ;
      }
      break ;

    case Pds::TransitionId::EndCalibCycle :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Running
        this->closeGroup( dgram, CalibCycle ) ;
      }

      break ;

    case Pds::TransitionId::L1Accept :

      // check the time, should not be sooner than begin of calib cycle
      if ( t < m_transClock[Pds::TransitionId::BeginCalibCycle] ) {
        MsgLog( logger, warning, "O2OHdf5Writer::eventStart: L1Accept time out of sync: "
              << Pds::TransitionId::name(dgram.seq.service())
              << " BeginCalibCycle time=" << m_transClock[Pds::TransitionId::BeginCalibCycle].toString("S%s%f")
              << " L1Accept time=" << t.toString("S%s%f")) ;
        skip = true;
      }

      break ;

    case Pds::TransitionId::Enable :
    case Pds::TransitionId::Disable :
    case Pds::TransitionId::Unknown :
    case Pds::TransitionId::Reset :
    case Pds::TransitionId::Map :
    case Pds::TransitionId::Unmap :
    case Pds::TransitionId::NumberOf :

      break ;
  }

  // store the time of the transition
  m_transClock[m_transition] = t;
  
  MsgLog( logger, debug, "O2OHdf5Writer -- now in the state " << ::stateName(m_state.top()) ) ;
  
  return not skip;
}

void
O2OHdf5Writer::eventEnd ( const Pds::Dgram& dgram )
{
}


void
O2OHdf5Writer::openGroup ( const Pds::Dgram& dgram, State state )
{
  // get the counter for this state
  unsigned counter = m_stateCounters[state] ;
  ++ m_stateCounters[state] ;

  // reset counter for sub-states, note there are no breaks
  switch( state ) {
  case Undefined:
    m_stateCounters[Configured] = 0;
  case Configured:
    m_stateCounters[Running] = 0;
  case Running:
    m_stateCounters[CalibCycle] = 0;
  case CalibCycle:
  case NumberOfStates:
    break;
  }

  // create group
  const std::string& name = groupName ( state, counter ) ;
  MsgLog( logger, debug, "HDF5Writer -- creating group " << name ) ;
  hdf5pp::Group group;
  if (m_groups.top().hasChild(name)) {
    group = m_groups.top().openGroup(name) ;
  } else {
    group = m_groups.top().createGroup(name) ;
  }

  // store transition time as couple of attributes to this new group
  ::storeClock ( group, dgram.seq.clock(), "start" ) ;

  // switch to new state
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

// visit the data object in configure or begincalibcycle transitions
void
O2OHdf5Writer::configObject(const void* data, size_t size,
    const Pds::TypeId& typeId, const O2OXtcSrc& src)
{
  // for Configure and BeginCalibCycle transitions store config objects at Source level
  MsgLog( logger, debug, "O2OHdf5Writer: store config object "
      << src.top()
      << " name=" <<  Pds::TypeId::name(typeId.id())
      << " version=" <<  typeId.version() ) ;
  m_configStore.store(typeId, src.top(), data, size);
}

// visit the data object
void
O2OHdf5Writer::dataObject ( const void* data, size_t size,
    const Pds::TypeId& typeId, const O2OXtcSrc& src )
{
  // find this type in the converter map
  CvtMap::iterator it = m_cvtMap.find( typeId.value() ) ;
  if ( it != m_cvtMap.end() ) {

    do {

      DataTypeCvtPtr converter = it->second ;
      try {
        converter->convert( data, size, typeId, src, m_eventTime ) ;
      } catch (const O2OXTCSizeException& ex) {
        // on size mismatch print an error message but continue
        MsgLog(logger, error, ex.what());
      }

      ++ it ;

    } while ( it != m_cvtMap.end() and it->first == typeId.value() ) ;

  } else if (typeId.id() != Pds::TypeId::Id_EpicsConfig) {

    MsgLogRoot( error, "O2OHdf5Writer::dataObject -- unexpected type or version: "
                << Pds::TypeId::name(typeId.id()) << "/" << typeId.version() ) ;

  }

}

// Construct a group name
std::string
O2OHdf5Writer::groupName( State state, unsigned counter ) const
{
  const char* prefix = "Undefined" ;
  switch ( state ) {
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
    case O2OHdf5Writer::NumberOfStates:
      break ;
  }

  if ( m_extGroups ) {
    // dump seconds as a hex string, it will be group name
    char buf[32] ;
    snprintf ( buf, sizeof buf, "%s:%04d", prefix, counter ) ;
    return buf;
  } else {
    return prefix;
  }
}

void
O2OHdf5Writer::openFile()
{
  std::string fileTempl = m_nameFactory.makePath ( m_split == Family ? O2OFileNameFactory::Family : 1 ) ;
  MsgLog( logger, debug, "O2OHdf5Writer - open output file " << fileTempl ) ;

  // Disable printing of error messages
  //stat = H5Eset_auto2( H5E_DEFAULT, 0, 0 ) ;

  // we want to create new file
  hdf5pp::PListFileAccess fapl ;
  if ( m_split == Family ) {
    // use FAMILY driver
    fapl.set_family_driver ( m_splitSize, hdf5pp::PListFileAccess() ) ;
  }

  // change the size of the B-Tree for chunked datasets
  hdf5pp::PListFileCreate fcpl;
  fcpl.set_istore_k(2);
  fcpl.set_sym_k(2, 2);

  hdf5pp::File::CreateMode mode = m_overwrite ? hdf5pp::File::Truncate : hdf5pp::File::Exclusive ;
  m_file = hdf5pp::File::create ( fileTempl, mode, fcpl, fapl ) ;

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

}

// close file/move to the final dir
void 
O2OHdf5Writer::closeFile()
{
  m_file.close();
  
  if (not m_finalDir.empty()) {
    // also move the file to the final destination directory
    if (m_split == Family) {
      
      // Family driver could have produced several files, find and move all of them  
      
      const std::string srcPattern = m_nameFactory.makePath(O2OFileNameFactory::FamilyPattern);
      fs::path srcDir = fs::path(srcPattern).parent_path();
      boost::regex pathRe(srcPattern);
      
      // iterate over files in a directory
      MsgLog(logger, info, "moving files from directory " << srcDir);
      for (fs::directory_iterator it(srcDir); it != fs::directory_iterator(); ++ it) {

        const fs::path src = it->path();
        if (boost::regex_match(src.string(), pathRe)) {

          fs::path basename = src.filename();
          fs::path dst = fs::path(m_finalDir) / basename;
          boost::system::error_code ec;
          MsgLog(logger, info, "renaming file " << src << " to " << dst);
          fs::rename(src, dst, ec);
          if (ec) {
            MsgLog( logger, error, "failed to rename file '" << src << "' to '" << dst << "': " << ec.message());
          }

        }
      }
      
    } else {
      
      // there should be just one file
      fs::path src = m_nameFactory.makePath(1);
      fs::path basename = src.filename();
      fs::path dst = fs::path(m_finalDir) / basename;
      boost::system::error_code ec;
      MsgLog(logger, info, "renaming file " << src << " to " << dst);
      fs::rename(src, dst, ec);
      if (ec) {
        MsgLog( logger, error, "failed to rename file '" << src << "' to '" << dst << "': " << ec.message());
      }

    }
  }
}

} // namespace O2OTranslator
