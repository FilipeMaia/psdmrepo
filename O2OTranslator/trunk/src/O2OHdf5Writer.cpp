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
#include <map>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <uuid/uuid.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "LusiTime/Time.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/O2OFileNameFactory.h"
#include "O2OTranslator/O2OMetaData.h"
#include "pdsdata/bld/bldData.hh"
#include "pdsdata/control/ConfigV1.hh"
#include "pdsdata/control/ConfigV2.hh"
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

  const char logger[] = "O2OTranslator.O2OHdf5Writer" ;

  // printable state name
  std::ostream& operator<<(std::ostream& out, O2OHdf5Writer::State state ) {
    const char* name = "*ERROR*" ;
    switch ( state ) {
      case O2OHdf5Writer::Undefined :
        name = "Undefined" ;
      case O2OHdf5Writer::Configured :
        name = "Configured" ;
      case O2OHdf5Writer::Running :
        name = "Running" ;
      case O2OHdf5Writer::CalibCycle :
        name = "CalibCycle" ;
      case O2OHdf5Writer::NumberOfStates :
        break ;
    }
    return out << name;
  }

  // store time as attributes to the group
  void storeClock ( hdf5pp::Group group, const Pds::ClockTime& clock, const std::string& what )
  {
    hdf5pp::Attribute<uint32_t> attr1 = group.createAttr<uint32_t> ( what+".seconds" ) ;
    attr1.store ( clock.seconds() ) ;
    hdf5pp::Attribute<uint32_t> attr2 = group.createAttr<uint32_t> ( what+".nanoseconds" ) ;
    attr2.store ( clock.nanoseconds() ) ;
  }


  // schema version number for the file structure
  int _fileSchemaVersion = 2;
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
                               bool extGroups,
                               const CvtOptions& cvtOptions,
                               const O2OMetaData& metadata,
                               const std::string& finalDir,
                               const std::string& backupExt,
                               bool fullTimeStamp )
  : O2OXtcScannerI()
  , m_nameFactory(nameFactory)
  , m_overwrite(overwrite)
  , m_split(split)
  , m_splitSize(splitSize)
  , m_extGroups(extGroups)
  , m_metadata(metadata)
  , m_finalDir(finalDir)
  , m_backupExt(backupExt)
  , m_fullTimeStamp(fullTimeStamp)
  , m_file()
  , m_state()
  , m_groups()
  , m_eventTime()
  , m_stateCounters()
  , m_transition(Pds::TransitionId::Unknown)
  , m_configStore0()
  , m_configStore()
  , m_calibStore()
  , m_cvtFactory(m_configStore, m_calibStore, m_metadata, cvtOptions)
  , m_transClock()
  , m_reopen(true)
  , m_serialScan(0)
  , m_scanAtOpen(0)
{
  std::fill_n(m_stateCounters, int(NumberOfStates), 0U);
  std::fill_n(m_transClock, int(Pds::TransitionId::NumberOf), LusiTime::Time(0,0));
    
  // we are in bad state, this state should never be popped
  m_state.push(Undefined) ;

  H5DataTypes::XtcClockTimeStamp::storeFullTimeStamp(fullTimeStamp);
}

//--------------
// Destructor --
//--------------
O2OHdf5Writer::~O2OHdf5Writer ()
{
  MsgLog(logger, debug, "O2OHdf5Writer - close output file");
  closeFile();
}

// signal start/end of the event (datagram)
bool
O2OHdf5Writer::eventStart ( const Pds::Dgram& dgram )
{
  MsgLog(logger, debug, "O2OHdf5Writer::eventStart " << Pds::TransitionId::name(dgram.seq.service())
          << " dgram.seq.type=" << dgram.seq.type()
          << " dgram.seq.service=" << Pds::TransitionId::name(dgram.seq.service()));

  m_transition = dgram.seq.service();
  Pds::ClockTime clock = dgram.seq.clock();
  LusiTime::Time t(clock.seconds(), clock.nanoseconds());
  
  // store current event time
  m_eventTime = H5DataTypes::XtcClockTimeStamp(clock, dgram.seq.stamp()) ;
  

  bool skip = false;
  switch ( m_transition ) {

    case Pds::TransitionId::Configure :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Undefined
        this->closeGroup( dgram, CalibCycle ) ;
        this->closeGroup( dgram, Running ) ;
        this->closeGroup( dgram, Configured ) ;
        if (m_reopen) {
          closeFile();
          openFile();
          m_reopen = false;
        }
        this->openGroup( dgram, Configured ) ;
                
        // reset content of the special configuration store
        m_configStore0.clear();
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
        if (m_reopen) {
          this->closeGroup( dgram, Configured, false ) ;
          closeFile();
          openFile();
          this->openGroup( dgram, Configured ) ;
          storeConfig0();
          m_reopen = false;
        }
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
        if (m_reopen) {
          this->closeGroup( dgram, Running, false ) ;
          this->closeGroup( dgram, Configured, false ) ;
          closeFile();
          openFile();
          this->openGroup( dgram, Configured ) ;
          storeConfig0();
          this->openGroup( dgram, Running ) ;
          m_reopen = false;
        }
        this->openGroup( dgram, CalibCycle ) ;
      }
      break ;

    case Pds::TransitionId::EndCalibCycle :

      if ( t != m_transClock[m_transition] ) {
        // close all states up to Running
        this->closeGroup( dgram, CalibCycle ) ;
        m_reopen = m_split == SplitScan;
        ++ m_serialScan;
      }

      break ;

    case Pds::TransitionId::L1Accept :

      // check the time, should not be sooner than begin of calib cycle
      if ( t < m_transClock[Pds::TransitionId::BeginCalibCycle] ) {
        MsgLog(logger, warning, "O2OHdf5Writer::eventStart: L1Accept time out of sync: "
              << Pds::TransitionId::name(dgram.seq.service())
              << " BeginCalibCycle time=" << m_transClock[Pds::TransitionId::BeginCalibCycle].toString("S%s%f")
              << " L1Accept time=" << t.toString("S%s%f"));
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
  
  MsgLog(logger, debug, "O2OHdf5Writer -- now in the state " << m_state.top()) ;
  
  return not skip;
}

void
O2OHdf5Writer::eventEnd ( const Pds::Dgram& dgram )
{
}


void
O2OHdf5Writer::openGroup ( const Pds::Dgram& dgram, State state)
{
  // create group
  const std::string& name = groupName ( state, m_stateCounters[state] ) ;
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
}

void
O2OHdf5Writer::closeGroup ( const Pds::Dgram& dgram, State state, bool updateCounters )
{
  if ( m_state.top() != state ) return ;

  if (updateCounters) {
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
  }
  

  // store transition time as couple of attributes to this new group
  ::storeClock ( m_groups.top(), dgram.seq.clock(), "end" ) ;

  // close all converters
  m_cvtFactory.closeGroup(m_groups.top());

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
O2OHdf5Writer::configObject(const void* data, size_t size, const Pds::TypeId& typeId,
    const O2OXtcSrc& src, Pds::Damage damage)
{
  if (damage.bits() != 0) {
    // for configuration objects we ignore any kind of damaged data
    return;
  }

  // for Configure and BeginCalibCycle transitions store config objects
  MsgLog( logger, debug, "O2OHdf5Writer: store config object " << src.top()
      << " name=" <<  Pds::TypeId::name(typeId.id())
      << " version=" <<  typeId.version() ) ;
  std::vector<char> vdata((const char*)data, ((const char*)data)+size);
  m_configStore.store(typeId, src.top(), vdata);
  if (m_transition == Pds::TransitionId::Configure) {
    // update also special config store
    m_configStore0.store(typeId, src.top(), vdata);
  }
  
  // For Control Config objects try to get estimate of event count per calib cycle
  if (typeId.id() == Pds::TypeId::Id_ControlConfig) {
    unsigned events = 0;
    switch(typeId.version()) {
    case 1: {
      const Pds::ControlData::ConfigV1& cfg = *static_cast<const Pds::ControlData::ConfigV1*>(data);
      if (cfg.uses_events()) events = cfg.events();
      break;
    }
    case 2: {
      const Pds::ControlData::ConfigV2& cfg = *static_cast<const Pds::ControlData::ConfigV2*>(data);
      if (cfg.uses_events()) events = cfg.events();
      break;
    }
    default:
      MsgLog(logger, warning, "Unexpected version of ControlData::Config object, chunk size estimate will not work");
      break;
    }

    if (events) {
      // there will be fluctuations in actual event count, try to put some upper limit (5% plus 10 events)
      hsize_t chunk = events + events / 20 + 10;
      CvtOptions::setChunkSize(chunk);
    }
  }
}

// store all configuration object from special store to a file
void
O2OHdf5Writer::storeConfig0() 
{
  for (ConfigObjectStore::const_iterator it = m_configStore0.begin(); it != m_configStore0.end(); ++ it) {
    
    const Pds::TypeId& typeId = it->first.first;
    O2OXtcSrc src;
    src.push(it->first.second);
    const std::vector<char>& data = it->second;
    
    dataObject(data.data(), data.size(), typeId, src, Pds::Damage(0));
    
  }
}

// visit the data object
void
O2OHdf5Writer::dataObject(const void* data, size_t size, const Pds::TypeId& typeId,
    const O2OXtcSrc& src, Pds::Damage damage)
{
  if (size == 0) {
    // sometimes happens
    MsgLog(logger, error, "O2OHdf5Writer::dataObject -- zero payload size: "
        << Pds::TypeId::name(typeId.id()) << "/" << typeId.version());
    return;
  }

  // Special case for Shared BLD data.
  if (typeId.id() == Pds::TypeId::Id_SharedIpimb or
      typeId.id() == Pds::TypeId::Id_SharedPim or
      typeId.id() == Pds::TypeId::Id_SharedAcqADC) {
    splitSharedObject(data, size, typeId, src, damage);
    return;
  }

  // find this type in the converter map, there is no converter for EpicsConfig,
  // it is handled internally by regular epics converter
  const O2OCvtFactory::DataTypeCvtList& cvts = m_cvtFactory.getConverters(m_groups.top(), typeId, src.top());
  if (cvts.empty() and typeId.id() != Pds::TypeId::Id_EpicsConfig) {
    MsgLog(logger, error, "O2OHdf5Writer::dataObject -- unexpected type or version: "
        << Pds::TypeId::name(typeId.id()) << "/" << typeId.version());
  }

  // call convert method on every matching converter
  typedef O2OCvtFactory::DataTypeCvtList::const_iterator iter;
  for (iter it = cvts.begin(); it != cvts.end(); ++ it) {
    try {
      (*it)->convert(data, size, typeId, src, m_eventTime, damage);
    } catch (const O2OXTCSizeException& ex) {
      // on size mismatch print an error message but continue
      MsgLog(logger, error, ex.what());
    }
  }

}


/*
 * Special case for Shared BLD data. We split them into their individual
 * components and store them as regular objects instead of one large
 * composite object. Components include both configuration and event data
 * objects so we update config store here as well.
 */
void
O2OHdf5Writer::splitSharedObject(const void* fulldata, size_t fullsize, const Pds::TypeId& typeId, const O2OXtcSrc& src, Pds::Damage damage)
{

  MsgLog( logger, debug, "O2OHdf5Writer:splitSharedObject -- splitting " << Pds::TypeId::name(typeId.id()) << "/" << typeId.version()) ;

  // When doing splitting special care needed for determining the size of the
  // split objects. When data is damaged (and sometimes when it is not) the size
  // of the XTC container can be different from the size of the data it should
  // contain. This is why we do gymnastics with sizes and offsets below.

  const char* data = static_cast<const char*>(fulldata);

  if (typeId.id() == Pds::TypeId::Id_SharedIpimb and typeId.version() == 0) {

    // config object should be stored first as it may be needed by data object
    Pds::TypeId typeId = Pds::TypeId(Pds::TypeId::Id_IpimbConfig, 1);
    size_t size = std::min(fullsize, sizeof(Pds::Ipimb::ConfigV1));
    this->configObject(data, size, typeId, src, damage);
    this->dataObject(data, size, typeId, src, damage);

    data += size;
    fullsize -= size;
    size = std::min(fullsize, sizeof(Pds::Ipimb::DataV1));
    typeId = Pds::TypeId(Pds::TypeId::Id_IpimbData, 1);
    this->dataObject(data, size, typeId, src, damage);

    data += size;
    fullsize -= size;
    size = fullsize;
    typeId = Pds::TypeId(Pds::TypeId::Id_IpmFex, 1);
    this->dataObject(data, size, typeId, src, damage);

  } else if (typeId.id() == Pds::TypeId::Id_SharedIpimb and typeId.version() == 1) {


    // config object should be stored first as it may be needed by data object
    Pds::TypeId typeId = Pds::TypeId(Pds::TypeId::Id_IpimbConfig, 2);
    size_t size = std::min(fullsize, sizeof(Pds::Ipimb::ConfigV2));
    this->configObject(data, size, typeId, src, damage);
    this->dataObject(data, size, typeId, src, damage);

    data += size;
    fullsize -= size;
    size = std::min(fullsize, sizeof(Pds::Ipimb::DataV2));
    typeId = Pds::TypeId(Pds::TypeId::Id_IpimbData, 2);
    this->dataObject(data, size, typeId, src, damage);

    data += size;
    fullsize -= size;
    size = fullsize;
    typeId = Pds::TypeId(Pds::TypeId::Id_IpmFex, 1);
    this->dataObject(data, size, typeId, src, damage);

  } else if (typeId.id() == Pds::TypeId::Id_SharedPim and typeId.version() == 1) {

    Pds::TypeId typeId;

    // config object should be stored first as it may be needed by data object
    typeId = Pds::TypeId(Pds::TypeId::Id_TM6740Config, 2);
    size_t size = std::min(fullsize, sizeof(Pds::Pulnix::TM6740ConfigV2));
    this->configObject(data, size, typeId, src, damage);
    this->dataObject(data, size, typeId, src, damage);

    data += size;
    fullsize -= size;
    size = std::min(fullsize, sizeof(Pds::Lusi::PimImageConfigV1));
    typeId = Pds::TypeId(Pds::TypeId::Id_PimImageConfig, 1);
    this->configObject(data, size, typeId, src, damage);
    this->dataObject(data, size, typeId, src, damage);

    data += size;
    fullsize -= size;
    size = fullsize;
    typeId = Pds::TypeId(Pds::TypeId::Id_Frame, 1);
    this->dataObject(data, size, typeId, src, damage);

  } else {

    MsgLog(logger, error, "O2OHdf5Writer::splitSharedObject -- unexpected type or version of shared BLD data: "
        << Pds::TypeId::name(typeId.id()) << "/" << typeId.version());

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
  std::string fileTempl = m_nameFactory.makePath ( m_split == Family ? O2OFileNameFactory::Family : m_serialScan ) ;
  m_scanAtOpen = m_serialScan;
  MsgLog( logger, trace, "O2OHdf5Writer - open output file " << fileTempl ) ;

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

  // store schema version for this file
  m_file.createAttr<uint32_t>(":schema:version").store(::_fileSchemaVersion);

  // add attributes specifying schema features
  const char* tsFormat = m_fullTimeStamp ? "full" : "short";
  m_file.createAttr<const char*>(":schema:timestamp-format").store(tsFormat) ;
  m_file.createAttr<uint32_t> (":schema:bld-shared-split").store(1);
  m_file.createAttr<uint32_t> (":schema:bld-config-as-evt").store(1);

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
  if (not m_file.valid()) {
    // it was not open
    return;
  }

  // close the top group first then file
  m_groups = std::stack<hdf5pp::Group>();
  m_file.close();
  
  if (not m_finalDir.empty()) {
    
    // also move the file to the final destination directory
    std::map<fs::path, fs::path> moveMap;
    
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
          fs::path dst = fs::path(m_finalDir) / src.filename();
          moveMap[src] = dst;
        }
      }
      
    } else {
      
      // there should be just one file
      fs::path src = m_nameFactory.makePath(m_scanAtOpen);
      fs::path basename = src.filename();
      fs::path dst = fs::path(m_finalDir) / src.filename();
      moveMap[src] = dst;
      
    }
    
    // check if the output files exists, try to backup them if m_backupExt is set
    for (std::map<fs::path, fs::path>::const_iterator it = moveMap.begin(); it != moveMap.end(); ++ it) {
      // try to backup existing file
      if (not m_backupExt.empty() and fs::exists(it->second)) {
        fs::path backup = it->second.string() + m_backupExt;
        MsgLog(logger, info, "backing up file " << it->second << " to " << backup);
        boost::system::error_code ec;
        fs::rename(it->second, backup, ec);
        if (ec) {
          MsgLog( logger, error, "failed to backup file '" << it->second << "' to '" << backup << "': " << ec.message());
        }
      }
      // check again
      if (fs::exists(it->second)) {
        MsgLog( logger, error, "cannot move file " << it->first << ", destination already exists: " << it->second);
        // reset whole thing
        moveMap.clear();
        break;
      }
    }
    
    // if anything left in the map move that
    for (std::map<fs::path, fs::path>::const_iterator it = moveMap.begin(); it != moveMap.end(); ++ it) {
      MsgLog(logger, info, "renaming file " << it->first << " to " << it->second);
      boost::system::error_code ec;
      fs::rename(it->first, it->second, ec);
      if (ec) {
        MsgLog( logger, error, "failed to rename file '" << it->first << "' to '" << it->second << "': " << ec.message());
      }
    }
    
  }

}

} // namespace O2OTranslator
