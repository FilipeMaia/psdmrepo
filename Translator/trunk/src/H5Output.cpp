#include <iostream>
#include <sstream>
#include <exception>
#include <algorithm>
#include <iterator>

#include <uuid/uuid.h>

#include "hdf5/hdf5.h"
#include "hdf5pp/Exceptions.h"
#include "ErrSvc/Issue.h"

#include "boost/make_shared.hpp"

#include "LusiTime/Time.h"

#include "MsgLogger/MsgLogger.h"
#include "PSEvt/DamageMap.h"
#include "PSEvt/TypeInfoUtils.h"

#include "psddl_pds2psana/cspad.ddl.h" // TODO: change pds2psana to psana, and all other places

#include "psddl_hdf2psana/DefaultChunkPolicy.h"

#include "Translator/H5Output.h"
#include "Translator/H5GroupNames.h"
#include "Translator/doNotTranslate.h"


using namespace Translator;
using namespace std;

namespace {

  int _fileSchemaVersion = 4; 

  string logger(string endLogger="") {
    const static string baseLogger = "Translator.H5Output"; 
    return ( endLogger.size()==0 ? baseLogger : baseLogger + string(".") + endLogger);
  }
    
  
  ///////////////////////////////////////////
  // initialization/config store functions

  /// takes a line like ' argument   # comment'  and returns 'argument'
  ///   that is it strips the comments, and then the line of any initial and trailing whitespace
  std::string stripComment(std::string s) {
    std::string::size_type idx = s.find('#');
    if (idx != std::string::npos) s = s.substr(0,idx);
    std::string::iterator pos = s.end();
    while (pos != s.begin()) {
      --pos;
      if (*pos != ' ') break;
    }
    return std::string(s.begin(),pos+1);
  }

  /// strips s of comments and trailing space, then it is include returns true, if
  ///   exclude returns false.  If anything else, causes a fatel error.  key is used in error message.
  bool excludeIncludeToBool(std::string key, std::string s) {
    static const std::string include = "include";
    static const std::string exclude = "exclude";
    s = stripComment(s);
    if (s == include) return true;
    if (s == exclude) return false;
    MsgLog(logger(),fatal,"parameter " << key << " is neither 'include' nor 'exclude', it is " << s);
    return false;
  }

  /// Takes a set of types to remove from the converter map.  Helper function for filtering
  /// types to translate.
  void removeTypes(HdfWriterMap  & hdfWriters, const TypeAliases::TypeInfoSet & typesToRemove) {
    TypeAliases::TypeInfoSet::const_iterator removePos;
    HdfWriterMap::iterator converterPos;
    for (removePos = typesToRemove.begin(); removePos != typesToRemove.end(); ++removePos) {
      const std::type_info *typeInfoPtr = *removePos;
      converterPos = hdfWriters.find(typeInfoPtr);
      if (converterPos == hdfWriters.end()) {
        MsgLog(logger(),fatal,"removeTypes: trying to remove " << PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr) 
               << " but type is not in converter list");
      }
      converterPos->second = boost::shared_ptr<HdfWriterBase>();
    }
  }

  string damageMapSummary(boost::shared_ptr<PSEvt::DamageMap> damageMap) {
    if (not damageMap) return string("null damage map");
    std::ostringstream str;
    map<uint32_t,int> damageCounts = damageMap->damageCounts();
    if (damageCounts.size()==0) str << "It is empty.";
    else {
      vector<uint32_t> damageValues;
      for (map<uint32_t,int>::iterator pos = damageCounts.begin();
           pos != damageCounts.end(); ++pos) {
        damageValues.push_back(pos->first);
      }
      sort(damageValues.begin(), damageValues.end());
      if (damageValues.size()==1 and damageValues[0]==0) {
        str << "Damage value is 0 for all data.";
      } else {
        str << "nonzero damage values:";
        str.setf(std::ios::hex,std::ios::basefield);
        if (damageValues.at(0) != 0) str << " " << damageValues.at(0);
        for (size_t idx=1; idx < damageValues.size(); ++idx) str << " " << damageValues.at(idx);
        str.unsetf(std::ios::hex);
      }
    }
    return str.str();
  }

  /////////// helper util //////////////
  // store time as attributes to the group
  void storeClock ( hdf5pp::Group &group, const PSTime::Time & clock, const std::string& what )
  {
    hdf5pp::Attribute<uint32_t> attr1 = group.createAttr<uint32_t> ( what+".seconds" ) ;
    attr1.store ( clock.sec() ) ;
    hdf5pp::Attribute<uint32_t> attr2 = group.createAttr<uint32_t> ( what+".nanoseconds" ) ;
    attr2.store ( clock.nsec() ) ;

    MsgLog(logger(),debug,"storeClock - stored " << what << ".seconds than .nanoseconds to group " << group);
  }

};

/////////////////////////////////////////
// constructor and initialization methods:

H5Output::H5Output(string moduleName) : Module(moduleName),
                                        m_currentConfigureCounter(0),
                                        m_currentRunCounter(0),
                                        m_currentCalibCycleCounter(0),
                                        m_currentEventCounter(0),
                                        m_filteredEventsThisCalibCycle(0),
                                        m_maxSavedPreviousSplitEvents(0),
                                        m_currentFilteredGroup(-1),
                                        m_event(0),
                                        m_env(0),
                                        m_totalConfigStoreUpdates(0),
                                        m_storeEpics(true)
{
  readConfigParameters();
  initializeHdfWriterMap(m_hdfWriters);
  filterHdfWriterMap();
  initializeCalibratedTypes();
  m_hdfWriterEventId = boost::make_shared<HdfWriterEventId>();
  m_hdfWriterDamage = boost::make_shared<HdfWriterDamage>();
  m_hdfWriterFilterMsg = boost::make_shared<HdfWriterFilterMsg>();
  m_hdfWriterEventId->setDatasetCreationProperties(m_eventIdCreateDsetProp);
  m_hdfWriterDamage->setDatasetCreationProperties(m_damageCreateDsetProp);
  m_hdfWriterFilterMsg->setDatasetCreationProperties(m_filterMsgCreateDsetProp);
  m_epicsGroupDir.initialize(m_hdfWriterEventId,
                             m_epicsPvCreateDsetProp);
  m_configureGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  m_calibCycleConfigureGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  m_calibCycleEventGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  initializeSrcFilter();
  openH5OutputFile();

  MsgLog(logger(),debug,name() << " constructor()");
}

void H5Output::readConfigParameters() {
  m_h5fileName = configStr("output_file");

  string splitStr = configStr("split","NoSplit");
  if (splitStr == "NoSplit") m_split = NoSplit;
  else if (splitStr == "Family") m_split = Family;
  else if (splitStr == "SplitScan") m_split = SplitScan;
  else MsgLog(logger(),fatal,"config parameter split must be one of 'NoSplit' 'Family' or 'SplitScan' (default is NoSplit)");

  m_splitSize = config("splitSize",10*1073741824ULL);

  // type filter parameters
  const set<string> & typeAliases = m_typeAliases.aliases();
  set<string>::const_iterator alias;
  for (alias = typeAliases.begin(); alias != typeAliases.end(); ++alias) {
    m_typeInclude[*alias] = excludeIncludeToBool(*alias,configStr(*alias,"include"));
  }
  m_storeEpics = config("store_epics",true);
  m_short_bld_name = config("short_bld_name",false);
  // src filter parameters
  std::list<std::string> include_all, empty_list;
  include_all.push_back("include");
  include_all.push_back("all");
  m_src_filter = configList("src_filter", include_all);

  // other translation parameters, ndarrays, calibration, metadata
  m_include_uncalibrated_data = config("include_uncalibrated_data",false);
  m_calibration_key = configStr("calibration_key","calibrated");

  // chunk parameters
  m_chunkSizeInBytes = config("chunkSizeInBytes",16*1024*1024);  // default - 16 MB chunks
  m_chunkSizeInElements = config("chunkSizeInElements", 0);
  m_maxChunkSizeInBytes = config("maxChunkSizeInBytes",100*1024*1024);
  m_minObjectsPerChunk = config("minObjectsPerChunk",50);
  m_maxObjectsPerChunk = config("maxObjectsPerChunk",2048);
  m_minChunkCacheSize = config("minChunkCacheSize",1024*1024);
  m_maxChunkCacheSize = config("maxChunkCacheSize",10*1024*1024);

  m_eventIdChunkSizeInBytes = config("eventIdChunkSizeInBytes",m_chunkSizeInBytes);
  m_eventIdChunkSizeInElements = config("eventIdChunkSizeInElements", m_chunkSizeInElements);

  m_damageChunkSizeInBytes = config("damageChunkSizeInBytes",m_chunkSizeInBytes);
  m_damageChunkSizeInElements = config("damageChunkSizeInElements",m_chunkSizeInBytes);

  m_filterMsgChunkSizeInBytes = config("filterMsgChunkSizeInBytes",m_chunkSizeInBytes);
  m_filterMsgChunkSizeInElements = config("filterMsgChunkSizeInElements",m_chunkSizeInElements);

  m_epicsPvChunkSizeInBytes = config("epicsPvChunkSizeBytes",m_chunkSizeInBytes);
  m_epicsPvChunkSizeInElements = config("epicsPvChunkSizeBytes",m_chunkSizeInElements);

  m_defaultChunkPolicy = boost::make_shared<psddl_hdf2psana::DefaultChunkPolicy>(
                                                             m_chunkSizeInBytes,
                                                             m_chunkSizeInElements,
                                                             m_maxChunkSizeInBytes,
                                                             m_minObjectsPerChunk,
                                                             m_maxObjectsPerChunk,
                                                             m_minChunkCacheSize,
                                                             m_maxChunkCacheSize);
  m_eventIdChunkPolicy   = boost::make_shared<psddl_hdf2psana::DefaultChunkPolicy>(
                                                             m_eventIdChunkSizeInBytes,
                                                             m_eventIdChunkSizeInElements,
                                                             m_maxChunkSizeInBytes,
                                                             m_minObjectsPerChunk,
                                                             m_maxObjectsPerChunk,
                                                             m_minChunkCacheSize,
                                                             m_maxChunkCacheSize);
  m_damageChunkPolicy    = boost::make_shared<psddl_hdf2psana::DefaultChunkPolicy>(
                                                             m_damageChunkSizeInBytes,
                                                             m_damageChunkSizeInElements,
                                                             m_maxChunkSizeInBytes,
                                                             m_minObjectsPerChunk,
                                                             m_maxObjectsPerChunk,
                                                             m_minChunkCacheSize,
                                                             m_maxChunkCacheSize);
  m_filterMsgChunkPolicy = boost::make_shared<psddl_hdf2psana::DefaultChunkPolicy>(
                                                             m_filterMsgChunkSizeInBytes,
                                                             m_filterMsgChunkSizeInElements,
                                                             m_maxChunkSizeInBytes,
                                                             m_minObjectsPerChunk,
                                                             m_maxObjectsPerChunk,
                                                             m_minChunkCacheSize,
                                                             m_maxChunkCacheSize);
  m_epicsPvChunkPolicy   = boost::make_shared<psddl_hdf2psana::DefaultChunkPolicy>(
                                                             m_epicsPvChunkSizeInBytes,
                                                             m_epicsPvChunkSizeInElements,
                                                             m_maxChunkSizeInBytes,
                                                             m_minObjectsPerChunk,
                                                             m_maxObjectsPerChunk,
                                                             m_minChunkCacheSize,
                                                             m_maxChunkCacheSize);

  m_defaultShuffle = config("shuffle",true);
  m_defaultDeflate = config("deflate",3);

  bool eventIdShuffle = config("eventIdShuffle",m_defaultShuffle);
  int eventIdDeflate = config("eventIdDeflate",m_defaultDeflate);

  bool damageShuffle = config("damageShuffle",m_defaultShuffle);
  int damageDeflate = config("damageDeflate",m_defaultDeflate);

  bool filterMsgShuffle = config("filterMsgShuffle",m_defaultShuffle);  
  int filterMsgDeflate = config("filterMsgDeflate",m_defaultDeflate);

  bool epicsPvShuffle = config("epicsPvShuffle",m_defaultShuffle); 
  int epicsPvDeflate = config("epicsPvDeflate",m_defaultDeflate);

  m_eventIdCreateDsetProp = DataSetCreationProperties(m_eventIdChunkPolicy,
                                                      eventIdShuffle,
                                                      eventIdDeflate);
  m_damageCreateDsetProp = DataSetCreationProperties(m_damageChunkPolicy,
                                                     damageShuffle,
                                                     damageDeflate);
  m_filterMsgCreateDsetProp = DataSetCreationProperties(m_filterMsgChunkPolicy,
                                                        filterMsgShuffle,
                                                        filterMsgDeflate);
  m_epicsPvCreateDsetProp = DataSetCreationProperties(m_epicsPvChunkPolicy,
                                                      epicsPvShuffle,
                                                      epicsPvDeflate);
  m_defaultCreateDsetProp = DataSetCreationProperties(m_defaultChunkPolicy,
                                                      m_defaultShuffle,
                                                      m_defaultDeflate);

  m_maxSavedPreviousSplitEvents = config("max_saved_split_events", 3000);
}

void H5Output::filterHdfWriterMap() {
  const TypeAliases::Alias2TypesMap & alias2TypesMap = m_typeAliases.alias2TypesMap();
  TypeAliases::Alias2TypesMap::const_iterator aliasPos;
  for (aliasPos = alias2TypesMap.begin(); aliasPos != alias2TypesMap.end(); ++aliasPos) {
    string alias = aliasPos->first;
    const TypeAliases::TypeInfoSet & typeInfoSet = aliasPos->second;
    if (not m_typeInclude[alias]) {
      removeTypes(m_hdfWriters, typeInfoSet);
    }
  }
}

void H5Output::initializeSrcFilter() {
  if (m_src_filter.size() < 2) MsgLog(logger(),fatal,"src_filter must start with either\n 'include' or 'exclude' and be followed by at least one src (which can be 'all' for include)");
  list<string>::iterator pos = m_src_filter.begin();
  string srcFilter0 = *pos;
  ++pos;
  string srcFilter1 = *pos;
  string include = "include";
  string exclude = "exclude";
  string all = "all";
  if (srcFilter0 != include and srcFilter0 != exclude) MsgLog(logger(), fatal, "src_filter first entry must be either 'include' or 'exclude'");
  m_srcFilterIsExclude = srcFilter0 == exclude;
  if (m_srcFilterIsExclude) {
    if (srcFilter1 == all) MsgLog(logger(), fatal, "src_filter = cannot be 'exclude all' this does no processing");
  } else {
    if (srcFilter1 == all) {
      m_includeAllSrc = true;
      MsgLog(logger(),debug,"initializeSrcFilter: include all");
      return;
    }
  }
  m_includeAllSrc = false;
  m_srcNameFilterSet.clear();
  while (pos != m_src_filter.end()) {
    m_srcNameFilterSet.insert(*pos);
    ++pos;
  }  
  WithMsgLog(logger(),debug,str) {
    str << "initializeSrcFilter: is_exclude=" 
        <<  m_srcFilterIsExclude << " filterSet: ";
    copy(m_srcNameFilterSet.begin(), m_srcNameFilterSet.end(),
         ostream_iterator<string>(str,", "));
  }
}

/// we need to update this set when new types come in that will
/// be replaced with calibrated data
void H5Output::initializeCalibratedTypes() {
  m_calibratedTypes.insert( & typeid(Psana::CsPad::DataV1) );
  m_calibratedTypes.insert( & typeid(Psana::CsPad::DataV2) );
  // TODO: make complete list of types that we look for calibrated data
}

void H5Output::openH5OutputFile() {
  unsigned majnum, minnum, relnum;
  herr_t err = H5get_libversion(&majnum, &minnum, &relnum);
  if (err != 0) throw hdf5pp::Hdf5CallException(ERR_LOC,"failed to get Hdf5 library version number");
  MsgLog(logger(),debug,"Hdf Library version info: " << majnum << "." << minnum << "." << relnum);

  hdf5pp::File::CreateMode mode = hdf5pp::File::Truncate;
  // change the size of the B-Tree for chunked datasets
  hdf5pp::PListFileCreate fcpl;
  fcpl.set_istore_k(2);
  fcpl.set_sym_k(2, 2);
  
  // we want to create new file
  hdf5pp::PListFileAccess fapl ;
  if ( m_split == Family ) {
    // use FAMILY driver
    fapl.set_family_driver ( m_splitSize, hdf5pp::PListFileAccess() ) ;
  }
  
  m_h5file = hdf5pp::File::create(m_h5fileName, mode, fcpl, fapl);

  // store schema version for this file
  m_h5file.createAttr<uint32_t>(":schema:version").store(::_fileSchemaVersion);

  // add attributes specifying schema features
  const char* tsFormat = "full"; // older translator supported a "short" format
  m_h5file.createAttr<const char*>(":schema:timestamp-format").store(tsFormat) ;
  m_h5file.createAttr<uint32_t> (":schema:bld-shared-split").store(1);
  m_h5file.createAttr<uint32_t> (":schema:bld-config-as-evt").store(1);

  // add UUID to the file attributes
  uuid_t uuid ;
  uuid_generate( uuid );
  char uuid_buf[64] ;
  uuid_unparse ( uuid, uuid_buf ) ;
  m_h5file.createAttr<const char*> ("UUID").store ( uuid_buf ) ;

  // add some metadata to the top group
  LusiTime::Time ctime = LusiTime::Time::now() ;
  m_h5file.createAttr<const char*> ("origin").store ( "psana-translator" ) ;
  m_h5file.createAttr<const char*> ("created").store ( ctime.toString().c_str() ) ;

}

void H5Output::setEventVariables(Event &evt, Env &env) {
  m_event = &evt;
  m_env = &env;
  m_eventId = evt.get();  // note, there is no eventId in the event at the end of the job
  setDamageMapFromEvent();
}

//////////////////////////////////////////////////////////
// Event Processing processing - Module methods

void H5Output::beginJob(Event& evt, Env& env) 
{
  MsgLog(logger(),trace,name() << ": beginJob()");
  setEventVariables(evt,env);

  // record some info from the env
  m_h5file.createAttr<uint32_t> ("expNum").store ( env.expNum() ) ;
  m_h5file.createAttr<const char*> ("experiment").store ( env.experiment().c_str() ) ;
  m_h5file.createAttr<const char*> ("instrument").store ( env.instrument().c_str() ) ;
  m_h5file.createAttr<const char*> ("jobName").store ( env.jobName().c_str() ) ;

  m_currentRunCounter = 0;
  createNextConfigureGroup();
  m_configureGroupDir.clearMaps();
  addConfigTypes(m_configureGroupDir, m_currentConfigureGroup);
  if (m_storeEpics) {
    m_epicsGroupDir.processBeginJob(m_currentConfigureGroup.id(), 
                                    env.epicsStore(), m_eventId);
  }
}

void H5Output::beginRun(Event& evt, Env& env) 
{
  MsgLog(logger(),debug,name() << ": beginRun()");
  setEventVariables(evt,env);
  m_currentCalibCycleCounter = 0;
  createNextRunGroup();  
}

void H5Output::beginCalibCycle(Event& evt, Env& env) 
{
  MsgLog(logger(),debug,name() << ": beginCalibCycle()");
  setEventVariables(evt,env);
  createNextCalibCycleGroup();
  m_calibCycleEventGroupDir.clearMaps();
  m_calibCycleConfigureGroupDir.clearMaps();
  m_currentEventCounter = 0;
  m_filteredEventsThisCalibCycle = 0;
  addConfigTypes(m_calibCycleConfigureGroupDir, m_currentCalibCycleGroup);
  if (m_storeEpics) {
    m_epicsGroupDir.processBeginCalibCycle(m_currentCalibCycleGroup.id(), env.epicsStore());
  }
}

void H5Output::event(Event& evt, Env& env) 
{
  setEventVariables(evt,env);
  try {
    eventImpl();
  } catch (...) {
    MsgLog(logger(),error,name() << "event: error, closing file");
    closeH5File();
    throw;
  }
  ++m_currentEventCounter;
}

void H5Output::setDamageMapFromEvent() {
  m_damageMap = m_event->get();
  if (not m_damageMap) {
    MsgLog(logger(),trace, eventPosition() << " No DamageMap in event");
    m_damageMap = boost::make_shared<PSEvt::DamageMap>();
  } else {
    WithMsgLog(logger(),trace, str) {
      str << eventPosition() << " DamageMap found. ";
      str << damageMapSummary(m_damageMap);
    }
    MsgLog(logger(), debug, "all entries in damageMap: " << *m_damageMap);
  }
}

Pds::Damage H5Output::getDamageForEventKey(const EventKey &eventKey) {
  Pds::Damage damage(0);
  PSEvt::DamageMap::iterator damagePos = m_damageMap->find(eventKey);
  if (damagePos != m_damageMap->end()) damage = damagePos->second.value();
  return damage;
}

/// see's if eventKey has been filtered through the source or type filters.
/// optionally checks if it should be skipped in lieu of a calibrated key.
//  The calibration check should be done for event data, but probably not for config
//  data.  If checkForCalibratedKey is true, a calibrated key is looked for in m_event.
//  Returns a null HdfWriter if:
//             there is no writer for this type
//             the src for the eventKey is filtered
//             the type for the eventKey is filtered
//             a calibrated version of the key exists (and we are not storing uncalibrated data)

boost::shared_ptr<HdfWriterBase> H5Output::checkTranslationFilters(const EventKey &eventKey, 
                                                                   bool checkForCalibratedKey) {
  const type_info * typeInfoPtr = eventKey.typeinfo();
  const Pds::Src & src = eventKey.src();
  const string & strKey = eventKey.key();
  boost::shared_ptr<HdfWriterBase> nullHdfWriter;
  if (doNotTranslate(src)) {
    MsgLog(logger(),debug,"doNotTranslate(src)==True for " << eventKey << " filtering");
    return nullHdfWriter;
  }
  boost::shared_ptr<HdfWriterBase> hdfWriter = getHdfWriter(m_hdfWriters, typeInfoPtr);
  if (not hdfWriter) {
    MsgLog(logger(),debug,"No hdfwriter found for type: " << PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr));
    return nullHdfWriter;
  }
  if (not m_include_uncalibrated_data and checkForCalibratedKey) {
    if (strKey.size()==0) {

      bool typeThatIsCalibrated = 
        m_calibratedTypes.find( typeInfoPtr ) != m_calibratedTypes.end();

      if (typeThatIsCalibrated) {
        EventKey calibratedKey(typeInfoPtr,src, m_calibration_key);
        if (m_event->proxyDict()->exists(calibratedKey)) {
          MsgLog(logger(), debug, "calibrated key exists for " << eventKey << " filtering");
          return nullHdfWriter;
        }
      }
    }
  }
  return hdfWriter;
}

list<EventKey> H5Output::getUpdatedConfigKeys() {
  list<EventKey> updatedConfigKeys;
  const PSEvt::HistI * configHist  = m_env->configStore().proxyDict()->hist();
  if (not configHist) MsgLog(logger("getUpdatedConfigKeys"),fatal,"Internal error - no HistI object in configStore");
  if (configHist->totalUpdates() <= m_totalConfigStoreUpdates) {
    return updatedConfigKeys;
  }
  m_totalConfigStoreUpdates = configHist->totalUpdates();
  list<EventKey> configKeys = m_env->configStore().keys();
  list<EventKey>::iterator iter;
  for (iter = configKeys.begin(); iter != configKeys.end(); ++iter) {
    EventKey &eventKey = *iter;
    long updates = configHist->updates(eventKey);
    map<EventKey, long>::iterator processedPos = m_configStoreUpdates.find(eventKey);
    if ( (processedPos == m_configStoreUpdates.end()) or 
         (processedPos->second < updates) ) {
      updatedConfigKeys.push_back(eventKey);
      m_configStoreUpdates[eventKey] = updates;
    }
  }
  return updatedConfigKeys;
}

list<EventKeyTranslation> H5Output::setEventKeysToTranslate(bool checkForCalibratedKey) {
  const string toAdd("setEventKeysToTranslate");
  list<EventKeyTranslation> toTranslate;

  list<EventKey> eventKeysFromEvent = m_event->keys();
  WithMsgLog(logger(toAdd), debug, str) {
    str << toAdd << " eventKeysFromEvent:";
    for (list<EventKey>::iterator pos = eventKeysFromEvent.begin();
         pos != eventKeysFromEvent.end(); ++pos) {
      str << " " << *pos;
    }
  }
  list<EventKey> updatedConfigKeys = getUpdatedConfigKeys();
  WithMsgLog(logger(toAdd), debug, str) {
    str << toAdd << " updated config keys:";
    for (list<EventKey>::iterator pos = updatedConfigKeys.begin();
         pos != updatedConfigKeys.end(); ++pos) {
      str << " " << *pos;
    }
  }
  set<EventKey> nonBlanks;

  typedef enum {ConfigList=0, EventList=1} ListType;
  for (int listType = ConfigList; listType <= EventList; ++listType) {
    list<EventKey> &eventKeys = updatedConfigKeys;
    DataTypeLoc dataTypeLoc = inConfigStore;
    if (listType == EventList) {
      eventKeys = eventKeysFromEvent;
      dataTypeLoc = inEvent;
    }
    list<EventKey>::iterator keyIter;
    for (keyIter = eventKeys.begin(); keyIter != eventKeys.end(); ++keyIter) {

      boost::shared_ptr<HdfWriterBase> hdfWriter = checkTranslationFilters(*keyIter, 
                                                                        checkForCalibratedKey);
      if (not hdfWriter) continue;
      Pds::Damage damage = getDamageForEventKey(*keyIter);
      toTranslate.push_back(EventKeyTranslation(*keyIter, damage, hdfWriter,
                                                EventKeyTranslation::NonBlank, 
                                                dataTypeLoc));
      nonBlanks.insert(*keyIter);
    }
  }
  // now add data that should get blanks due to damage
  PSEvt::DamageMap::iterator damagePos;
  for (damagePos = m_damageMap->begin(); damagePos != m_damageMap->end(); ++damagePos) {
    Pds::Damage damage = damagePos->second;
    if (damage.value()==0) continue;
    const EventKey &eventKey = damagePos->first;
    bool alreadyAddedAsNonBlank = nonBlanks.find(eventKey) != nonBlanks.end();
    if (alreadyAddedAsNonBlank) continue;

    boost::shared_ptr<HdfWriterBase> hdfWriter = checkTranslationFilters(eventKey, 
                                                                         checkForCalibratedKey);
    if (not hdfWriter) continue;
    toTranslate.push_back(EventKeyTranslation(eventKey,damage,hdfWriter,
                                              EventKeyTranslation::Blank,
                                              inEvent));
  }
  WithMsgLog(logger(toAdd),trace,str) {
    str << "checkForCalib= " << checkForCalibratedKey << " EventKeyTranslation list: ";
    list<EventKeyTranslation>::iterator pos;
    for (pos = toTranslate.begin(); pos != toTranslate.end(); ++pos) {
      EventKeyTranslation & eventKeyTranslation = *pos;
      str << eventKeyTranslation;
      str << ", ";
    }
  }
  return toTranslate;
}

void H5Output::addToFilteredEventDataset(const PSEvt::EventId &eventId, const std::string & msg) {
  if (m_filteredEventsThisCalibCycle==0) {
    hid_t calibGroup = m_currentCalibCycleGroup.id();
    m_currentFilteredGroup = H5Gcreate(calibGroup, "filtered",
                            H5P_DEFAULT, // propery list for link creation
                            H5P_DEFAULT, // group creation
                            H5P_DEFAULT);  // group access, not implemented in hdf5
    if (m_currentFilteredGroup < 0) throw hdf5pp::Hdf5CallException(ERR_LOC,"Could not create filtered group");
    m_hdfWriterEventId->make_dataset(m_currentFilteredGroup);
    m_hdfWriterFilterMsg->make_dataset(m_currentFilteredGroup);
  }
  m_hdfWriterEventId->append(m_currentFilteredGroup, eventId);
  m_hdfWriterFilterMsg->append(m_currentFilteredGroup, msg);
  ++m_filteredEventsThisCalibCycle;
}

bool H5Output::checkForAndProcessExcludeEvent() {
  boost::shared_ptr<Translator::ExcludeEvent> excludeEvent = m_event->get();
  if (excludeEvent) {
    addToFilteredEventDataset(*m_eventId,excludeEvent->getMsg());
    MsgLog(logger(),trace,"exclude event found " << eventPosition());
    return true;
  }
  return false;
}

void H5Output::eventImpl() 
{
  static const string eventImpl("eventImpl");
  bool filteredEvent = checkForAndProcessExcludeEvent();
  if (filteredEvent) return;
  if (m_storeEpics) {
    m_epicsGroupDir.processEvent(m_env->epicsStore(), m_eventId);
  }
  list<EventKeyTranslation> toTranslate = setEventKeysToTranslate(true);
  vector<pair<Pds::Src,Pds::Damage> > droppedContribs = m_damageMap->getSrcDroppedContributions();
  bool splitEvent = droppedContribs.size()>0;
  bool repeatEvent = false;
  map<EventKey, long, LessEventKey> previousBlanksForRepeatSplitEvent;
  set<EventKey, LessEventKey> previousNonBlanksForRepeatSplitEvent;
  if (splitEvent) {
    MsgLog(logger(eventImpl),trace,"split event " << eventPosition());
    BlankNonBlanksMap::iterator pos = m_previousSplitEvents.find(m_eventId);
    repeatEvent = pos != m_previousSplitEvents.end();
    if (repeatEvent) {
      MsgLog(logger(eventImpl),trace,"split event AND repeat event");
      previousBlanksForRepeatSplitEvent = pos->second.blanks;
      previousNonBlanksForRepeatSplitEvent = pos->second.nonblanks;
    }
  }

  m_calibCycleEventGroupDir.markAllSrcKeyGroupsNotWrittenForEvent();
  MsgLog(logger(eventImpl),trace,eventPosition() << " eventId: " << *m_eventId);
  list<EventKeyTranslation>::iterator keyIter;
  for (keyIter = toTranslate.begin(); keyIter != toTranslate.end(); ++keyIter) {
    const EventKey eventKey = keyIter->eventKey;
    bool writeBlank = keyIter->entryType == EventKeyTranslation::Blank;
    DataTypeLoc dataTypeLoc = keyIter->dataTypeLoc;
    Pds::Damage damage = keyIter->damage;
    boost::shared_ptr<HdfWriterBase> hdfWriter = keyIter->hdfWriter;
    const std::type_info * typeInfoPtr = eventKey.typeinfo();
    const Pds::Src & src = eventKey.src();
    MsgLog(logger(),debug,eventImpl << " eventKey=" << eventKey << "damage= " << damage.value() << 
           " writeBlank=" << writeBlank << " loc=" << dataTypeLoc << " hdfwriter=" << hdfWriter);
    TypeMapContainer::iterator typePos = m_calibCycleEventGroupDir.findType(typeInfoPtr);
    if (typePos == m_calibCycleEventGroupDir.endType()) {
      m_calibCycleEventGroupDir.addTypeGroup(typeInfoPtr, m_currentCalibCycleGroup, m_short_bld_name);
      MsgLog(logger(eventImpl),trace,
             PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr) <<" not in groups.  Added type to groups");
    }
    SrcKeyMap::iterator srcKeyPos = m_calibCycleEventGroupDir.findSrcKey(eventKey);
    if (srcKeyPos == m_calibCycleEventGroupDir.endSrcKey(typeInfoPtr)) {
      SrcKeyGroup & srcKeyGroup = m_calibCycleEventGroupDir.addSrcKeyGroup(eventKey,hdfWriter);
      MsgLog(logger(eventImpl),trace,
             "src " << src << " not in type group.  Added src to type group");
      if (writeBlank) {
        MsgLog(logger(eventImpl),trace," initial event is blank.  Only creating time/damage datasets");
        srcKeyGroup.make_timeDamageDatasets();
      } else {
        MsgLog(logger(eventImpl),trace," initial event is nonblank.  Creating event datasets, and time/damage datasets");
        srcKeyGroup.make_datasets(dataTypeLoc, *m_event, *m_env, m_defaultCreateDsetProp);
      }
      srcKeyPos = m_calibCycleEventGroupDir.findSrcKey(eventKey);
    }
    SrcKeyGroup & srcKeyGroup = srcKeyPos->second;
    bool needToAppend = true;
    if (repeatEvent) {
      map<PSEvt::EventKey, long>::iterator previousBlank;
      previousBlank = previousBlanksForRepeatSplitEvent.find(eventKey);
      if (previousBlank != previousBlanksForRepeatSplitEvent.end()) {
        MsgLog(logger(eventImpl),trace,
               "repeat event has blank entry for eventKey " << eventKey);
        size_t blankIndex = previousBlank->second;
        if (writeBlank) srcKeyGroup.overwriteDamage(blankIndex, eventKey, m_eventId, damage);
        else srcKeyGroup.overwriteDataAndDamage(blankIndex, eventKey, dataTypeLoc, *m_event, *m_env, damage);
        srcKeyGroup.written(true);
        needToAppend = false;
      } else {
        MsgLog(logger(eventImpl),trace,
               "repeat event has nonblank entry for eventKey " << eventKey);
        set<PSEvt::EventKey>::iterator previousNonBlank;
        previousNonBlank = previousNonBlanksForRepeatSplitEvent.find(eventKey);
        if (previousNonBlank != previousNonBlanksForRepeatSplitEvent.end()) {
          needToAppend = false;
          // we do not expect this to happend, print warning
          WithMsgLog(logger(eventImpl),warning,str) {
            str << "repeatEvent at " << eventPosition() << " eventKey= " << eventKey;
            str << " previous entry *has* data, it is non blank, unexpected.";
            str << " Current entry is: ";
            if (writeBlank) {
              str  << "nonblank";
            } else {
              str << "blank";
            }
            str << ". Ignoring current entry.";
          }
        }
      }
    }
    if (needToAppend) {
      if (writeBlank) {
        MsgLog(logger(),debug,eventImpl << " appending blank");
        srcKeyGroup.appendBlankTimeAndDamage(eventKey, m_eventId, damage);
      }
      else {
        if (not srcKeyGroup.arrayTypeDatasetsCreated()) {
          MsgLog(logger(eventImpl),trace, "appending nonblank and type dataset not created, making type datasets");
          srcKeyGroup.make_typeDatasets(dataTypeLoc, *m_event, *m_env, m_defaultCreateDsetProp);
        } else {
          MsgLog(logger(),debug,eventImpl << " appending nonblank");
        }
        srcKeyGroup.appendDataTimeAndDamage(eventKey, dataTypeLoc, *m_event, *m_env, m_eventId, damage);
      }
      srcKeyGroup.written(true);
    }
  }
  
  if (splitEvent and not repeatEvent) {
    // As a given source can have several types of data, there may be two or more 
    // droppedContributions from a given source.  That is there may be Src repeats in the 
    // droppedContribs list below.  If there is at least one dropped contribution from a
    // source, we will append blanks to all the types from that source that were not already written.
    // The damage we use for these blanks will be from one of these droppedContrib entries with
    // that Src. We have no way to identify which damage will go with which type in this case.
    // This situtation will be rare, and when it happens, most likely all droppedContrib entries
    // from the same Src will have the same damage - the DroppedContribution bit.

    BlankNonBlanks blankNonBlanks;
    set<Pds::Src> droppedSrcs;
    map<Pds::Src,Pds::Damage> src2damage;
    for (size_t idx = 0; idx < droppedContribs.size(); ++idx) {
      Pds::Src src = droppedContribs[idx].first;
      Pds::Damage damage = droppedContribs[idx].second;
      droppedSrcs.insert(src);
      src2damage[src]=damage;
    }
    map<Pds::Src, vector<PSEvt::EventKey> > droppedSrcsNotWritten;
    vector<PSEvt::EventKey> otherNotWritten;
    vector<PSEvt::EventKey> writtenKeys;
    m_calibCycleEventGroupDir.getNotWrittenSrcPartition(droppedSrcs,
                                                 droppedSrcsNotWritten,
                                                 otherNotWritten,
                                                 writtenKeys);
    set<Pds::Src>::iterator droppedSrc;
    for (droppedSrc = droppedSrcs.begin(); droppedSrc != droppedSrcs.end(); ++droppedSrc) {
      const Pds::Src & src = *droppedSrc;
      map<Pds::Src, vector<PSEvt::EventKey> >::iterator notWrittenIter;
      notWrittenIter= droppedSrcsNotWritten.find(src);
      if (notWrittenIter == droppedSrcsNotWritten.end()) {
        MsgLog(logger(eventImpl),warning, 
               "dropped src: " << src << " has not been seen before.  Not writing a blank.");
        continue;
      }
      vector<PSEvt::EventKey> & notWritten = notWrittenIter->second;
      for (size_t notWrittenIdx = 0; notWrittenIdx < notWritten.size(); ++notWrittenIdx) {
        PSEvt::EventKey & eventKey = notWritten[notWrittenIdx];
        SrcKeyMap::iterator srcKeyPos = m_calibCycleEventGroupDir.findSrcKey(eventKey);
        SrcKeyGroup & srcKeyGroup = srcKeyPos->second;
        Pds::Damage damage = src2damage[src];
        long pos = srcKeyGroup.appendBlankTimeAndDamage(eventKey, m_eventId, damage);
        blankNonBlanks.blanks[eventKey] = pos;
      }
    }
    for (size_t writtenIdx = 0; writtenIdx < writtenKeys.size(); ++writtenIdx)  {
      PSEvt::EventKey eventKey = writtenKeys[writtenIdx];
      blankNonBlanks.nonblanks.insert(eventKey); 
    }
    if (m_previousSplitEvents.size() < m_maxSavedPreviousSplitEvents) {
      m_previousSplitEvents[m_eventId]=blankNonBlanks;
    } else {
      WithMsgLog(logger(eventImpl),warning,str) {
        str << "Maximum number of cached splitEvents reached. ";
        str << "Will not be able to fill in blanks for this event: " << *m_eventId;
      }
    }
  }
}

void H5Output::addConfigTypes(TypeSrcKeyH5GroupDirectory &configGroupDirectory,
                              hdf5pp::Group & parentGroup) {
  const string addTo("addConfigureTypes");
  list<EventKey> envEventKeys = getUpdatedConfigKeys();
  list<EventKey> evtEventKeys = m_event->keys();
  int newTypes=0;
  int newSrcs=0;
  int newDatasets=0;
  for (int locIdx = 0; locIdx < 2; ++locIdx) {
    DataTypeLoc dataLoc = inConfigStore;
    bool checkForCalib = false;
    list<EventKey> &eventKeys = envEventKeys;
    list<EventKey>::iterator iter;
    if (locIdx == 1) {
      eventKeys = evtEventKeys;
      dataLoc = inEvent;
      checkForCalib = true;  // unlikely that there is a calib key for a config type, we
                             // can probably safely keep checkForCalib as false
    }
    for (iter = eventKeys.begin(); iter != eventKeys.end(); ++iter) {
      EventKey &eventKey = *iter;
      MsgLog(logger(addTo),debug,"addConfigureTypes eventKey: " << *iter << " loc: " << dataLoc);
      boost::shared_ptr<HdfWriterBase> hdfWriter = checkTranslationFilters(eventKey,checkForCalib);
      if (not hdfWriter) continue;
      const std::type_info * typeInfoPtr = eventKey.typeinfo();
      const Pds::Src & src = eventKey.src();
      TypeMapContainer::iterator typePos = configGroupDirectory.findType(typeInfoPtr);
      if (typePos == configGroupDirectory.endType()) {
        ++newTypes;
        configGroupDirectory.addTypeGroup(typeInfoPtr, parentGroup, m_short_bld_name);
        MsgLog(logger(addTo),trace, 
               PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr) <<" not in groups.  Added type to groups");
      }
      SrcKeyMap::iterator srcKeyPos = configGroupDirectory.findSrcKey(eventKey);
      if (srcKeyPos == configGroupDirectory.endSrcKey(typeInfoPtr)) {
        ++newSrcs;
        configGroupDirectory.addSrcKeyGroup(eventKey,hdfWriter);
        MsgLog(logger(addTo), trace,
               " src " << src << " not in type group.  Added src to type group");
        srcKeyPos = configGroupDirectory.findSrcKey(eventKey);
      }
      SrcKeyGroup & srcKeyGroup = srcKeyPos->second;
      if (dataLoc == inConfigStore) {
        try {
          srcKeyGroup.storeData(eventKey, inConfigStore, *m_event, *m_env);
        } catch (ErrSvc::Issue &issue) {
          configGroupDirectory.dump();
          throw issue;
        }
      } else if (dataLoc == inEvent) {
        srcKeyGroup.make_datasets(inEvent, *m_event, *m_env, m_defaultCreateDsetProp);
        Pds::Damage damage = getDamageForEventKey(eventKey);
        srcKeyGroup.appendDataTimeAndDamage(eventKey, inEvent, *m_event, *m_env, m_eventId, damage);
      }
      ++newDatasets;
    }
  }
  MsgLog(logger(addTo), trace, "created " << newTypes << " new types, " << newSrcs << " new srcs, and "
         << newDatasets << " newDatasets");
}

void H5Output::endCalibCycle(Event& evt, Env& env) {
  MsgLog(logger(),trace,"endCalibCycle()");
  setEventVariables(evt,env);
  if (m_filteredEventsThisCalibCycle>0) {
    m_hdfWriterFilterMsg->closeDataset(m_currentFilteredGroup);
    m_hdfWriterEventId->closeDataset(m_currentFilteredGroup);
    herr_t err = H5Gclose(m_currentFilteredGroup);
    if (err<0) MsgLog(logger(),fatal,"Failed to close current filtered group: " << m_currentFilteredGroup);
  }
  m_currentFilteredGroup = -1;
  if (m_storeEpics) m_epicsGroupDir.processEndCalibCycle();
  m_calibCycleEventGroupDir.closeGroups();
  m_calibCycleConfigureGroupDir.closeGroups();
  if (m_eventId) ::storeClock ( m_currentCalibCycleGroup, m_eventId->time(), "end" ) ;
  m_currentCalibCycleGroup.close();
  ++m_currentCalibCycleCounter;
}

void H5Output::endRun(Event& evt, Env& env) 
{
  MsgLog(logger(),trace,"endRun()");
  setEventVariables(evt,env);
  if (m_eventId) ::storeClock ( m_currentRunGroup, m_eventId->time(), "end" ) ;
  m_currentRunGroup.close();
  ++m_currentRunCounter;

}

void H5Output::endJob(Event& evt, Env& env) 
{
  setEventVariables(evt,env);
  m_configureGroupDir.closeGroups();
  if (m_storeEpics) m_epicsGroupDir.processEndJob();
  if (m_eventId) ::storeClock ( m_currentConfigureGroup, m_eventId->time(), "end" ) ;
  m_currentConfigureGroup.close();
  ++m_currentConfigureCounter;

  MsgLog(logger(),trace,name() << ": endJob()");
}

////////////////////////////////////////////////////
// shut down

H5Output::~H5Output() 
{
  m_h5file.close();
}

//////////////////////////////////////////////////////
// Helper functions to module event processing

void H5Output::createNextConfigureGroup() {
  char groupName[128];
  sprintf(groupName,"Configure:%4.4lu", m_currentConfigureCounter);
  m_currentConfigureGroup = m_h5file.createGroup(groupName);
  if (m_eventId) ::storeClock ( m_currentConfigureGroup, m_eventId->time(), "start" ) ;
  MsgLog(logger(),trace,name() << ": createNextConfigureGroup: " << groupName);
}

void H5Output::createNextRunGroup() {
  char groupName[128];
  sprintf(groupName,"Run:%4.4lu", m_currentRunCounter);
  m_currentRunGroup = m_currentConfigureGroup.createGroup(groupName);
  if (m_eventId) ::storeClock ( m_currentRunGroup, m_eventId->time(), "start" ) ;

  MsgLog(logger(),trace,name() << ": createNextRunGroup: " << groupName);
}

void H5Output::createNextCalibCycleGroup() {
  char groupName[128];
  sprintf(groupName,"CalibCycle:%4.4lu", m_currentCalibCycleCounter);
  m_currentCalibCycleGroup = m_currentRunGroup.createGroup(groupName);
  if (m_eventId) ::storeClock ( m_currentCalibCycleGroup, m_eventId->time(), "start" ) ;

  MsgLog(logger(),trace,name() << ": createNextCalibCycleGroup: " << groupName);
}

void H5Output::closeH5File() {
  m_calibCycleEventGroupDir.closeGroups();
  m_configureGroupDir.closeGroups();
  m_currentCalibCycleGroup.close();
  m_currentRunGroup.close();
  m_currentConfigureGroup.close();
  m_h5file.close();
}

// doNotTranslate logs a trace message when it filters a source, or returns true
bool H5Output::doNotTranslate(const Pds::Src &src) {
  if (m_includeAllSrc) return false;
  string srcName = getH5GroupNameForSrc(src);
  bool srcInFilterList = m_srcNameFilterSet.find(srcName) != m_srcNameFilterSet.end();
  if (m_srcFilterIsExclude) {
    bool srcListedInExcludeList = srcInFilterList;
    if (srcListedInExcludeList) MsgLog(logger(),trace,"doNotTranslate: src= " << src << " is in exclude list - no translation");
    return srcListedInExcludeList;
  } 
  // it is an include set
  bool srcNotListedInIncludeSet = not srcInFilterList;
  if (srcNotListedInIncludeSet) MsgLog(logger(),trace,"doNotTranslate: src=" << src << " is not listed in include list - no translation");
  return (srcNotListedInIncludeSet);
}

string H5Output::eventPosition() {
  stringstream res;
  res << "Configure:" << m_currentConfigureCounter
      << "/Run:" << m_currentRunCounter
      << "/CalibCycle:" << m_currentCalibCycleCounter
      << "/Event:" << m_currentEventCounter;
  return res.str();
}

PSANA_MODULE_FACTORY(H5Output);
