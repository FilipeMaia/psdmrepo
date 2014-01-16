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

#include "psddl_psana/cspad.ddl.h"   // for specifying calibrated types

#include "Translator/H5Output.h"
#include "Translator/H5GroupNames.h"
#include "Translator/doNotTranslate.h"
#include "Translator/HdfWriterNewDataFromEvent.h"

using namespace Translator;
using namespace std;

namespace {

  int _fileSchemaVersion = 4; 

  string logger(string endLogger="") {
    const static string baseLogger = "Translator.H5Output"; 
    return ( endLogger.size()==0 ? baseLogger : baseLogger + string(".") + endLogger);
  }
  
  bool isString( const std::type_info *typeInfoPtr) {
    if (*typeInfoPtr == typeid(std::string)) return true;
    return false;
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
      converterPos->second = boost::shared_ptr<HdfWriterFromEvent>();
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
                                        m_event(0),
                                        m_env(0),
                                        m_totalConfigStoreUpdates(-1),
                                        m_storeEpics(EpicsH5GroupDirectory::Unknown)
{
  MsgLog(logger(),trace,name() << " constructor()");
  readConfigParameters();
  initializeHdfWriterMap(m_hdfWriters);
  filterHdfWriterMap();
  initializeCalibratedTypes();
  m_hdfWriterEventId = boost::make_shared<HdfWriterEventId>();
  m_hdfWriterDamage = boost::make_shared<HdfWriterDamage>();
  m_hdfWriterEventId->setDatasetCreationProperties(m_eventIdCreateDsetProp);
  m_hdfWriterDamage->setDatasetCreationProperties(m_damageCreateDsetProp);
  m_epicsGroupDir.initialize(m_storeEpics,
                             m_hdfWriterEventId,
                             m_epicsPvCreateDsetProp);
  m_configureGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  m_calibCycleConfigureGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  m_calibCycleEventGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  boost::shared_ptr<HdfWriterDamage> nullHdfWriterDamage;
  m_calibCycleFilteredGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,nullHdfWriterDamage);
  TypeAliases::Alias2TypesMap::const_iterator pos = m_typeAliases.alias2TypesMap().find("ndarray_types");
  if (pos == m_typeAliases.alias2TypesMap().end()) MsgLog(logger(), fatal, "The TypeAliases map does not include ndarray_types as a key");
  if ((pos->second).size() == 0) MsgLog(logger(), fatal, "There are no types assigned to the  'ndarray_types' alias in the TypeAliases map");
  m_h5groupNames = boost::make_shared<H5GroupNames>(m_short_bld_name, pos->second);
  m_configureGroupDir.setH5GroupNames(m_h5groupNames);
  m_calibCycleConfigureGroupDir.setH5GroupNames(m_h5groupNames);
  m_calibCycleEventGroupDir.setH5GroupNames(m_h5groupNames);
  m_calibCycleFilteredGroupDir.setH5GroupNames(m_h5groupNames);
  openH5OutputFile();
}

void H5Output::readConfigParameters() {
  MsgLog(logger(), trace, "reading config parameters");
  m_h5fileName = configStr("output_file");
  string splitStr = configStr("split","NoSplit");
  if (splitStr == "NoSplit") m_split = NoSplit;
  else if (splitStr == "Family") m_split = Family;
  else if (splitStr == "SplitScan") m_split = SplitScan;
  else MsgLog(logger(),fatal,"config parameter 'split' must be one of 'NoSplit' 'Family' or 'SplitScan' (default is NoSplit)");

  m_splitSize = config("splitSize",10*1073741824ULL);

  // type filter parameters
  const set<string> & typeAliases = m_typeAliases.aliases();
  set<string>::const_iterator alias;
  for (alias = typeAliases.begin(); alias != typeAliases.end(); ++alias) {
    m_typeInclude[*alias] = excludeIncludeToBool(*alias,configStr(*alias,"include"));
  }
  map<string, EpicsH5GroupDirectory::EpicsStoreMode> validStoreEpicsInput;
  validStoreEpicsInput["no"]=EpicsH5GroupDirectory::DoNotStoreEpics;
  validStoreEpicsInput["calib_repeat"]=EpicsH5GroupDirectory::RepeatEpicsEachCalib;
  validStoreEpicsInput["updates_only"]=EpicsH5GroupDirectory::OnlyStoreEpicsUpdates;
  string storeEpics = configStr("store_epics", "updates_only");
  map<string, EpicsH5GroupDirectory::EpicsStoreMode>::iterator userInput = validStoreEpicsInput.find(storeEpics);
  if (userInput == validStoreEpicsInput.end()) {
    MsgLog(logger(), fatal, "config parameter 'epics_store' must be one of 'calib_repeat' 'updates_only' or 'no'. The value: '"
           << storeEpics << "' is invalid");
  }
  m_storeEpics = userInput->second;
  MsgLog(logger(),trace,"epics storage: " << EpicsH5GroupDirectory::epicsStoreMode2str(m_storeEpics));

  m_short_bld_name = config("short_bld_name",false);
  m_create_alias_links = config("create_alias_links",true);
  m_overwrite = config("overwrite",false);

  // src filter parameters
  std::list<std::string> include_all, empty_list;
  include_all.push_back("include");
  include_all.push_back("all");
  m_src_filter = configList("src_filter", include_all);

  // key filter parameters
  m_ndarray_key_filter = configList("ndarray_key_filter", include_all);
  m_std_string_key_filter = configList("std_string_key_filter", include_all);

  // other translation parameters, calibration, metadata
  m_include_uncalibrated_data = config("include_uncalibrated_data",false);
  m_calibration_key = configStr("calibration_key","calibrated");

  m_chunkManager.readConfigParameters(*this);

  m_defaultShuffle = config("shuffle",true);
  m_defaultDeflate = config("deflate",1);

  bool eventIdShuffle = config("eventIdShuffle",m_defaultShuffle);
  int eventIdDeflate = config("eventIdDeflate",m_defaultDeflate);

  bool damageShuffle = config("damageShuffle",false);
  int damageDeflate = config("damageDeflate",m_defaultDeflate);

  bool stringShuffle = config("stringShuffle",false);
  int stringDeflate = config("stringDeflate",-1);

  bool epicsPvShuffle = config("epicsPvShuffle",false);
  int epicsPvDeflate = config("epicsPvDeflate",m_defaultDeflate);

  bool ndarrayShuffle = config("ndarrayShuffle",m_defaultShuffle);
  int ndarrayDeflate = config("ndarrayDeflate",m_defaultDeflate);

  m_eventIdCreateDsetProp = DataSetCreationProperties(m_chunkManager.eventIdChunkPolicy(),
                                                      eventIdShuffle,
                                                      eventIdDeflate);
  m_damageCreateDsetProp = DataSetCreationProperties(m_chunkManager.damageChunkPolicy(),
                                                     damageShuffle,
                                                     damageDeflate);
  m_stringCreateDsetProp = DataSetCreationProperties(m_chunkManager.stringChunkPolicy(),
                                                     stringShuffle,
                                                     stringDeflate);
  m_epicsPvCreateDsetProp = DataSetCreationProperties(m_chunkManager.epicsPvChunkPolicy(),
                                                      epicsPvShuffle,
                                                      epicsPvDeflate);
  m_defaultCreateDsetProp = DataSetCreationProperties(m_chunkManager.defaultChunkPolicy(),
                                                      m_defaultShuffle,
                                                      m_defaultDeflate);
  m_ndarrayCreateDsetProp = DataSetCreationProperties(m_chunkManager.ndarrayChunkPolicy(),
                                                      ndarrayShuffle,
                                                      ndarrayDeflate);

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

namespace {
  void parseFilterConfigString(const string &configParamKey, 
                               const list<string> & configParamValues,
                               bool & isExclude,
                               bool & includeAll,
                               set<string> & filterSet) 
  {
    if (configParamValues.size() < 2) {
      MsgLog(logger(),fatal, configParamKey 
             << " must start with either\n 'include' or 'exclude' and be followed by at least one src (which can be 'all' for include)");
    }
    filterSet.clear();
    list<string>::const_iterator pos = configParamValues.begin();
    string filter0 = *pos;
    ++pos;
    string filter1 = *pos;
    string include = "include";
    string exclude = "exclude";
    string all = "all";
    if (filter0 != include and filter0 != exclude) {
      MsgLog(logger(), fatal, configParamKey << " first entry must be either 'include' or 'exclude'");
    }
    isExclude = filter0 == exclude;
    if (isExclude) {
      if (filter1 == all) MsgLog(logger(), fatal, "src_filter = cannot be 'exclude all' this does no processing");
    } else {
      if (filter1 == all) {
        includeAll = true;
        MsgLog(logger(),debug, configParamKey << ": include all");
        return;
      }
    }
    includeAll = false;
    filterSet.clear();
    while (pos != configParamValues.end()) {
      filterSet.insert(*pos);
      ++pos;
    }  
    WithMsgLog(logger(), debug, str) {
      str << configParamKey << ": is_exclude=" 
          << isExclude << " filterSet: ";
      copy(filterSet.begin(), filterSet.end(),
           ostream_iterator<string>(str,", "));
    }
   }

} // local namespace

void H5Output::initializeSrcAndKeyFilters() {
  set<string> srcNameFilterSet;
  parseFilterConfigString("src_filter", m_src_filter, m_srcFilterIsExclude, m_includeAllSrc, srcNameFilterSet);
  m_psevtSourceFilterList.clear();
  for (set<string>::iterator pos = srcNameFilterSet.begin(); pos != srcNameFilterSet.end(); ++pos) {
    PSEvt::Source source(*pos);
    PSEvt::Source::SrcMatch srcMatch = source.srcMatch(*(m_env->aliasMap()));
    m_psevtSourceFilterList.push_back(srcMatch);
  }
  MsgLog(logger(),trace, "src_filter: isExclude=" << m_srcFilterIsExclude << " all=" << m_includeAllSrc);
  parseFilterConfigString("ndarray_key_filter", m_ndarray_key_filter, m_ndarrayKeyIsExclude,   m_includeAllNdarrayKey,   m_ndarrayKeyFilterSet);
  parseFilterConfigString("std_string_key_filter",  m_std_string_key_filter,  m_stdStringKeyIsExclude, m_includeAllStdStringKey, m_stdStringKeyFilterSet);
}

// when updating the calibration set below, update the 
// calibration filtering section of default_psana.cfg with the
// complete, updated type list.
void H5Output::initializeCalibratedTypes() {
  m_calibratedTypes.insert( & typeid(Psana::CsPad::DataV1) );
  m_calibratedTypes.insert( & typeid(Psana::CsPad::DataV2) );
}

void H5Output::openH5OutputFile() {
  MsgLog(logger(),trace,name() << " opening h5 output file");
  unsigned majnum, minnum, relnum;
  herr_t err = H5get_libversion(&majnum, &minnum, &relnum);
  if (err != 0) throw hdf5pp::Hdf5CallException(ERR_LOC,"failed to get Hdf5 library version number");
  MsgLog(logger(),debug,"Hdf Library version info: " << majnum << "." << minnum << "." << relnum);

  hdf5pp::File::CreateMode mode = m_overwrite ? hdf5pp::File::Truncate : hdf5pp::File::Exclusive ;

  // change the size of the B-Tree for chunked datasets
  hdf5pp::PListFileCreate fcpl;
  fcpl.set_istore_k(2);
  fcpl.set_sym_k(2, 2);
  
  // we want to create new file
  hdf5pp::PListFileAccess fapl ;
  if ( m_split != NoSplit ) {
    MsgLog(logger(), fatal, "hdf5 splitting is not implemented.  Only NoSplit is presently supported");
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

void H5Output::checkForNewWriters() {
  list<EventKey> eventKeys = m_event->keys();
  list<EventKey>::iterator keyIter;
  for (keyIter = eventKeys.begin(); keyIter != eventKeys.end(); ++keyIter) {
    EventKey &eventKey = *keyIter;
    const std::type_info & keyCppType = *eventKey.typeinfo();
    if (keyCppType == typeid(Translator::HdfWriterNew)) {
      const Pds::Src &src = eventKey.src();
      const string &key = eventKey.key();
      boost::shared_ptr<Translator::HdfWriterNew> newWriter = m_event->get(src,key);
      if (not newWriter) {
        MsgLog(logger(),error," unexpected: could not retrieve new writer");
        return;
      }
      const std::type_info * newType = newWriter->typeInfoPtr();
      MsgLog(logger(),trace," new hdf5 writer found for type: "
             << PSEvt::TypeInfoUtils::typeInfoRealName(newType)
             << " key: " << key);
      boost::shared_ptr<HdfWriterNewDataFromEvent> newWriterFromEvent = 
        boost::make_shared<HdfWriterNewDataFromEvent>(*newWriter,key);
      // overwrite the old writer if it is there.
      if (m_hdfWriters.find(newType) != m_hdfWriters.end()) {
        MsgLog(logger(), warning, " overwriting previous writer");
      }
      m_hdfWriters[newType] = newWriterFromEvent;
    }
  }
}

//////////////////////////////////////////////////////////
// Event Processing processing - Module methods

void H5Output::beginJob(Event& evt, Env& env) 
{
  MsgLog(logger(),trace,"H5Output beginJob()");
  setEventVariables(evt,env);
  initializeSrcAndKeyFilters();
  if (m_create_alias_links) {
    m_configureGroupDir.setAliasMap(env.aliasMap());
    m_calibCycleConfigureGroupDir.setAliasMap(env.aliasMap());
    m_calibCycleEventGroupDir.setAliasMap(env.aliasMap());
    m_calibCycleFilteredGroupDir.setAliasMap(env.aliasMap());
  }

  // record some info from the env
  m_h5file.createAttr<uint32_t> ("expNum").store ( env.expNum() ) ;
  m_h5file.createAttr<const char*> ("experiment").store ( env.experiment().c_str() ) ;
  m_h5file.createAttr<const char*> ("instrument").store ( env.instrument().c_str() ) ;
  m_h5file.createAttr<const char*> ("jobName").store ( env.jobName().c_str() ) ;

  m_currentRunCounter = 0;
  checkForNewWriters();
  createNextConfigureGroup();
  m_configureGroupDir.clearMaps();
  m_chunkManager.beginJob(env);
  addConfigTypes(m_configureGroupDir, m_currentConfigureGroup);
  m_epicsGroupDir.processBeginJob(m_currentConfigureGroup.id(), 
                                  env.epicsStore(), m_eventId);
}

void H5Output::beginRun(Event& evt, Env& env) 
{
  MsgLog(logger(),debug,name() << ": beginRun()");
  setEventVariables(evt,env);
  // the aliasMap can change from run to run, so reinialize the src filter list with each run.
  initializeSrcAndKeyFilters();
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
  m_calibCycleFilteredGroupDir.clearMaps();
  m_currentEventCounter = 0;
  m_filteredEventsThisCalibCycle = 0;
  m_chunkManager.beginCalibCycle(env);
  addConfigTypes(m_calibCycleConfigureGroupDir, m_currentCalibCycleGroup);
  m_epicsGroupDir.processBeginCalibCycle(m_currentCalibCycleGroup.id(), env.epicsStore());
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

bool H5Output::isNDArray( const type_info *typeInfoPtr) {
  return m_h5groupNames->isNDArray(typeInfoPtr);
}

Pds::Damage H5Output::getDamageForEventKey(const EventKey &eventKey) {
  Pds::Damage damage(0);
  PSEvt::DamageMap::iterator damagePos = m_damageMap->find(eventKey);
  if (damagePos != m_damageMap->end()) damage = damagePos->second.value();
  return damage;
}

/// see's if eventKey has been filtered through the type, source, or key filters.
/// optionally checks if it should be skipped in lieu of a calibrated key.
//  The calibration check should be done for event data, but not for config
//  data.  If checkForCalibratedKey is true, a calibrated key is looked for in m_event.
//  Returns a null HdfWriter if:
//             there is no writer for this type
//             the src for the eventKey is filtered
//             the type for the eventKey is filtered
//             the key is for std::string or ndarray and the key for this EventKey is filtered
//             the type is one of the special calibrated types, a calibrated version 
//               of the key exists, and we are not storing uncalibrated data

boost::shared_ptr<HdfWriterFromEvent> H5Output::checkTranslationFilters(const EventKey &eventKey, 
                                                                   bool checkForCalibratedKey) {
  const type_info * typeInfoPtr = eventKey.typeinfo();
  const Pds::Src & src = eventKey.src();
  const string & strKey = eventKey.key();
  boost::shared_ptr<HdfWriterFromEvent> nullHdfWriter;
  if (srcIsFiltered(src)) {
    MsgLog(logger(),debug,"srcIsFiltered(src)==True for " << eventKey << " filtering");
    return nullHdfWriter;
  }
  if (isString(typeInfoPtr) and stringKeyIsFiltered(strKey)) return nullHdfWriter;
  if (isNDArray(typeInfoPtr) and ndarrayKeyIsFiltered(strKey)) return nullHdfWriter;

  boost::shared_ptr<HdfWriterFromEvent> hdfWriter = getHdfWriter(m_hdfWriters, typeInfoPtr);
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
  if (m_totalConfigStoreUpdates > -1 and (configHist->totalUpdates() <= m_totalConfigStoreUpdates)) {
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

void H5Output::setEventKeysToTranslate(list<EventKeyTranslation> & toTranslate, 
                                       list<EventKey> &filtered) {
  const string toAdd("setEventKeysToTranslate");
  toTranslate.clear();
  filtered.clear();
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
  list<EventKey>::iterator keyIter;
  for (keyIter = updatedConfigKeys.begin(); keyIter != updatedConfigKeys.end(); ++keyIter) {
    boost::shared_ptr<HdfWriterFromEvent> hdfWriter = checkTranslationFilters(*keyIter,false);
    if (not hdfWriter) continue;
    Pds::Damage damage = getDamageForEventKey(*keyIter);
    toTranslate.push_back(EventKeyTranslation(*keyIter, damage, hdfWriter,
                                              EventKeyTranslation::NonBlank, 
                                              inConfigStore));
    nonBlanks.insert(*keyIter);
  }

  bool eventIsFiltered = false;
  list<EventKeyTranslation> toTranslateFromEvent;
  for (keyIter = eventKeysFromEvent.begin(); keyIter != eventKeysFromEvent.end(); ++keyIter) {
    bool eventFilterKey = hasDoNotTranslatePrefix(keyIter->key());
    if (eventFilterKey) {
      filtered.push_back(*keyIter);
      eventIsFiltered = true;
      continue;
    }
    if (eventIsFiltered) continue;

    boost::shared_ptr<HdfWriterFromEvent> hdfWriter = checkTranslationFilters(*keyIter,true);
    if (not hdfWriter) continue;

    Pds::Damage damage = getDamageForEventKey(*keyIter);
    toTranslateFromEvent.push_back(EventKeyTranslation(*keyIter, damage, hdfWriter,
                                                       EventKeyTranslation::NonBlank, 
                                                       inEvent));
    nonBlanks.insert(*keyIter);
  }

  // if the event is filtered, still return updated config keys, but nothing from the
  // event for translation to the CalibCycle group - the filtered list will contain any
  // data to go to the filtered group
  if (eventIsFiltered) {
    WithMsgLog(logger(toAdd),trace,str) {
      str << " event is filtered, filtered key list: ";
      list<PSEvt::EventKey>::iterator pos;
      for (pos = filtered.begin(); pos != filtered.end(); ++pos) {
        str << *pos << ", ";
      }
    }
    WithMsgLog(logger(toAdd),trace,str) {
      str << " EventKeyTranslation list: ";
      list<EventKeyTranslation>::iterator pos;
      for (pos = toTranslate.begin(); pos != toTranslate.end(); ++pos) {
        EventKeyTranslation & eventKeyTranslation = *pos;
        str << eventKeyTranslation;
        str << ", ";
      }
    }
    return;
  }

  toTranslate.splice(toTranslate.end(), toTranslateFromEvent);

  // now add data that should get blanks due to damage
  PSEvt::DamageMap::iterator damagePos;
  for (damagePos = m_damageMap->begin(); damagePos != m_damageMap->end(); ++damagePos) {
    Pds::Damage damage = damagePos->second;
    if (damage.value()==0) continue;
    const EventKey &eventKey = damagePos->first;
    bool alreadyAddedAsNonBlank = nonBlanks.find(eventKey) != nonBlanks.end();
    if (alreadyAddedAsNonBlank) continue;

    boost::shared_ptr<HdfWriterFromEvent> hdfWriter = checkTranslationFilters(eventKey, true);
    if (not hdfWriter) continue;

    toTranslate.push_back(EventKeyTranslation(eventKey,damage,hdfWriter,
                                              EventKeyTranslation::Blank,
                                              inEvent));
  }
  WithMsgLog(logger(toAdd),trace,str) {
    str << " EventKeyTranslation list: ";
    list<EventKeyTranslation>::iterator pos;
    for (pos = toTranslate.begin(); pos != toTranslate.end(); ++pos) {
      EventKeyTranslation & eventKeyTranslation = *pos;
      str << eventKeyTranslation;
      str << ", ";
    }
  }
}

void H5Output::addToFilteredEventGroup(const list<EventKey> &eventKeys, const PSEvt::EventId &eventId) {
  if (m_filteredEventsThisCalibCycle==0) {
    char filteredGroupName[128];
    sprintf(filteredGroupName,"Filtered:%4.4lu", m_currentCalibCycleCounter);
    m_currentFilteredGroup = m_currentRunGroup.createGroup(filteredGroupName);
    m_hdfWriterEventId->make_dataset(m_currentFilteredGroup);
  }
  ++m_filteredEventsThisCalibCycle;
  m_hdfWriterEventId->append(m_currentFilteredGroup, eventId);
  list<EventKey>::const_iterator iter;
  for (iter = eventKeys.begin(); iter != eventKeys.end(); ++iter) {
    const PSEvt::EventKey &eventKey = *iter;
    boost::shared_ptr<HdfWriterFromEvent> hdfWriter = checkTranslationFilters(eventKey, true);
    if (not hdfWriter) continue;
    const std::type_info * typeInfoPtr = eventKey.typeinfo();
    TypeMapContainer::iterator typePos = m_calibCycleFilteredGroupDir.findType(typeInfoPtr);
    if (typePos == m_calibCycleFilteredGroupDir.endType()) {
      m_calibCycleFilteredGroupDir.addTypeGroup(typeInfoPtr, m_currentFilteredGroup);
    }
    SrcKeyMap::iterator srcKeyPos = m_calibCycleFilteredGroupDir.findSrcKey(eventKey);
    if (srcKeyPos == m_calibCycleFilteredGroupDir.endSrcKey(typeInfoPtr)) {
      SrcKeyGroup & srcKeyGroup = m_calibCycleFilteredGroupDir.addSrcKeyGroup(eventKey,hdfWriter);
      srcKeyGroup.make_datasets(inEvent, *m_event, *m_env, m_defaultCreateDsetProp);
      srcKeyPos = m_calibCycleFilteredGroupDir.findSrcKey(eventKey);
    }
    SrcKeyGroup & srcKeyGroup = srcKeyPos->second;
    Pds::Damage dummyDamage(0);
    srcKeyGroup.appendDataTimeAndDamage(eventKey, inEvent, *m_event, *m_env, m_eventId, dummyDamage);
  }
}

void H5Output::eventImpl() 
{
  static const string eventImpl("eventImpl");
  m_epicsGroupDir.processEvent(m_env->epicsStore(), m_eventId);
  list<EventKeyTranslation> toTranslate;
  list<EventKey> filtered;
  setEventKeysToTranslate(toTranslate, filtered);
  if (filtered.size()>0) {
    addToFilteredEventGroup(filtered, *m_eventId);
  }
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
    boost::shared_ptr<HdfWriterFromEvent> hdfWriter = keyIter->hdfWriter;
    const std::type_info * typeInfoPtr = eventKey.typeinfo();
    const Pds::Src & src = eventKey.src();
    MsgLog(logger(),debug,eventImpl << " eventKey=" << eventKey << "damage= " << damage.value() << 
           " writeBlank=" << writeBlank << " loc=" << dataTypeLoc << " hdfwriter=" << hdfWriter);
    TypeMapContainer::iterator typePos = m_calibCycleEventGroupDir.findType(typeInfoPtr);
    if (typePos == m_calibCycleEventGroupDir.endType()) {
      m_calibCycleEventGroupDir.addTypeGroup(typeInfoPtr, m_currentCalibCycleGroup);
      MsgLog(logger(eventImpl),trace, "type: " << PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr) 
             << " with group name " << m_h5groupNames->nameForType(typeInfoPtr)
             << " not in calibCycleEventGroupDir.  Added type to groups");
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
  const string addTo("addConfigTypes");
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
      checkForCalib = true;
    }
    for (iter = eventKeys.begin(); iter != eventKeys.end(); ++iter) {
      EventKey &eventKey = *iter;
      MsgLog(logger(addTo),debug,"addConfigureTypes eventKey: " << *iter << " loc: " << dataLoc);
      boost::shared_ptr<HdfWriterFromEvent> hdfWriter = checkTranslationFilters(eventKey,checkForCalib);
      if (not hdfWriter) continue;
      const std::type_info * typeInfoPtr = eventKey.typeinfo();
      const Pds::Src & src = eventKey.src();
      TypeMapContainer::iterator typePos = configGroupDirectory.findType(typeInfoPtr);
      if (typePos == configGroupDirectory.endType()) {
        ++newTypes;
        configGroupDirectory.addTypeGroup(typeInfoPtr, parentGroup);
        MsgLog(logger(addTo),trace, "type: " << PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr) 
               << " with group name " << m_h5groupNames->nameForType(typeInfoPtr)
               << " not in configGroupDir.  Added type to groups");
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
    m_hdfWriterEventId->closeDataset(m_currentFilteredGroup);
    m_calibCycleFilteredGroupDir.closeGroups();
    m_currentFilteredGroup.close();
  }
  m_epicsGroupDir.processEndCalibCycle();
  m_calibCycleEventGroupDir.closeGroups();
  m_calibCycleConfigureGroupDir.closeGroups();
  if (m_eventId) ::storeClock ( m_currentCalibCycleGroup, m_eventId->time(), "end" ) ;
  m_currentCalibCycleGroup.close();
  m_chunkManager.endCalibCycle(m_currentEventCounter);
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
  m_epicsGroupDir.processEndJob();
  if (m_eventId) ::storeClock ( m_currentConfigureGroup, m_eventId->time(), "end" ) ;
  m_currentConfigureGroup.close();
  m_chunkManager.endJob();
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

namespace {

  bool keyIsFiltered(const string & key, const bool includeAll, const bool isExclude, const set<string> & filterSet) {
    if (includeAll) return false;
    bool keyInFilterList = filterSet.find(key) != filterSet.end();
    if (isExclude) {
      bool inExcludeList = keyInFilterList;
      if (inExcludeList) {
        MsgLog(logger(),trace,"keyIsFiltered: key= " << key << " is in exclude list - no translation");
        return true;
      } else {
        return false;
      }
    } 
    // it is an include set
    bool keyNotInIncludeSet = not keyInFilterList;
    if (keyNotInIncludeSet) {
      MsgLog(logger(),trace,"keyIsFiltered: key=" << key << " is not listed in include list - no translation");
      return true;
    }
    return false;
  }

} // local namespace

bool H5Output::stringKeyIsFiltered(const string &key) {
  bool retVal = keyIsFiltered(key, m_includeAllStdStringKey, m_stdStringKeyIsExclude, m_stdStringKeyFilterSet);
  return retVal;
}

bool H5Output::ndarrayKeyIsFiltered(const string &key) {
  bool retVal = keyIsFiltered(key, m_includeAllNdarrayKey, m_ndarrayKeyIsExclude, m_ndarrayKeyFilterSet);
  return retVal;
}

bool H5Output::srcIsFiltered(const Pds::Src &src) {
  if (m_includeAllSrc) return false;
  bool srcInList = false;
  for (unsigned idx = 0; idx < m_psevtSourceFilterList.size(); ++idx) {
    if (m_psevtSourceFilterList.at(idx).match(src)) {
      srcInList = true;
      break;
    }
  }
  if (m_srcFilterIsExclude) {
    bool inExcludeList = srcInList;
    if (inExcludeList) {
      MsgLog(logger(),trace,"src: log=0x" << hex << src.log() << " phy=0x" << src.phy() 
             << " name=" << m_h5groupNames->nameForSrc(src) 
             << " matches Source exclude list - no translation");
      return true;
    } else {
      return false;
    }
  } 
  // it is an include set
  bool srcNotInIncludeSet = not srcInList;
  if (srcNotInIncludeSet) {
    MsgLog(logger(),trace,"src: log=0x" << hex << src.log() << " phy=0x" << src.phy() 
           << " name=" << m_h5groupNames->nameForSrc(src) 
           << " does not match any Source in include list - no translation");
    return true;
  }
  return false;
}

string H5Output::eventPosition() {
  stringstream res;
  res << "eventPosition: Configure:" << m_currentConfigureCounter
      << "/Run:" << m_currentRunCounter
      << "/CalibCycle:" << m_currentCalibCycleCounter
      << "/Event:" << m_currentEventCounter;
  return res.str();
}

PSANA_MODULE_FACTORY(H5Output);
