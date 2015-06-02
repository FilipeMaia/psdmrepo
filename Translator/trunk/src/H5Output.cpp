//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class H5Output
//     psana module class for translating xtc to hdf5
//
// Author List:
//     David Schneider
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Translator/H5Output.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <sstream>
#include <exception>
#include <algorithm>
#include <iterator>
#include "boost/make_shared.hpp"
#include <uuid/uuid.h>
#include "hdf5/hdf5.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Exceptions.h"
#include "ErrSvc/Issue.h"
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/DamageMap.h"
#include "PSEvt/TypeInfoUtils.h"
#include "Translator/H5GroupNames.h"
#include "Translator/specialKeyStrings.h"
#include "Translator/HdfWriterNewDataFromEvent.h"
#include "Translator/LoggerNameWithMpiRank.h"
#include "Translator/H5MpiSplitScanDefaults.h"
#include "PSEvt/Exceptions.h"

// headers for problem with translating alias list from both DAQ and Control streams
#include "pdsdata/xtc/ProcInfo.hh"
#include "psddl_psana/alias.ddl.h"

using namespace Translator;
using namespace std;

#define DBGLVL debug
#define TRACELVL trace

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {
  int _fileSchemaVersion = 5; 

  LoggerNameWithMpiRank logger("Translator.H5Output");
  
  bool isString( const std::type_info *typeInfoPtr) {
    if (*typeInfoPtr == typeid(std::string)) return true;
    return false;
  }

  bool psanaSkipedEvent(Event &evt) {
    return evt.exists<int>( "__psana_skip_event__");
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
    MsgLog(logger,fatal,"parameter " << key << " is neither 'include' nor 'exclude', it is " << s);
    return false;
  }

  /// Takes a set of types to remove from the converter map.  Helper function for filtering
  /// types to translate.
  void removeTypes(HdfWriterMap  & hdfWriters, const TypeAliases::TypeInfoSet & typesToRemove) {
    TypeAliases::TypeInfoSet::const_iterator removePos;
    for (removePos = typesToRemove.begin(); removePos != typesToRemove.end(); ++removePos) {
      const std::type_info *typeInfoPtr = *removePos;
      bool hadType = hdfWriters.remove(typeInfoPtr);
      if (not hadType) {
        MsgLog(logger,fatal,"removeTypes: trying to remove " << PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr) 
               << " but type is not in converter list");
      }
    }
  }

  /////////// helper util //////////////
  // store time as attributes to the group
  void storeClock ( hdf5pp::Group &group, const PSTime::Time & clock, const std::string& what )
  {
    hdf5pp::Attribute<uint32_t> attr1 = group.createAttr<uint32_t> ( what+".seconds" ) ;
    attr1.store ( clock.sec() ) ;
    hdf5pp::Attribute<uint32_t> attr2 = group.createAttr<uint32_t> ( what+".nanoseconds" ) ;
    attr2.store ( clock.nsec() ) ;

    MsgLog(logger,DBGLVL,"storeClock - stored " << what << ".seconds than .nanoseconds to group " << group);
  }

  void parseFilterConfigString(const string &configParamKey, 
                               const list<string> & configParamValues,
                               bool & isExclude,
                               bool & includeAll,
                               set<string> & filterSet,
                               bool excludeAllOk = false) 
  {
    if (configParamValues.size() < 2) {
      MsgLog(logger,fatal, configParamKey 
             << " must start with either\n 'include' or 'exclude' and be followed by at least one entry (which can be 'all' for include)");
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
      MsgLog(logger, fatal, configParamKey << " first entry must be either 'include' or 'exclude'");
    }
    isExclude = filter0 == exclude;
    if (isExclude) {
      if ((filter1 == all) and (not excludeAllOk)) {
        MsgLog(logger, fatal, configParamKey << " cannot be 'exclude all' this does no processing");
      }
    } else {
      if (filter1 == all) {
        includeAll = true;
        MsgLog(logger,DBGLVL, configParamKey << ": include all");
        return;
      }
    }
    includeAll = false;
    filterSet.clear();
    while (pos != configParamValues.end()) {
      filterSet.insert(*pos);
      ++pos;
    }  
    WithMsgLog(logger, DBGLVL, str) {
      str << configParamKey << ": is_exclude=" 
          << isExclude << " filterSet: ";
      copy(filterSet.begin(), filterSet.end(),
           ostream_iterator<string>(str,", "));
    }
   }

  void problemWithTranslatingBothDAQ_and_Control_AliasLists(list<EventKey> &eventKeys) {
    // presently the Alias list from the DAQ and Control streams occurs with
    // the 'Control' source differing only by the ip address, so they would end up in the
    // same dataset and the Translator will skip the second one. Here we try to get the 
    // DAQ one first on the assumption that it tends to have 0.0.0.0
    // for the IP address. This is a hack until we sort out the names for this type of src.
    list<EventKey>::iterator iter;
    bool foundDaqAliasKey = false;
    for (iter = eventKeys.begin(); iter != eventKeys.end(); ++iter) {
      if (*(iter->typeinfo()) == typeid(Psana::Alias::ConfigV1)) {
        Pds::Src src = iter->src();
        if (src.level() == Pds::Level::Control) {
          Pds::ProcInfo *procInfo = static_cast<const Pds::ProcInfo *>(&src);
          if (procInfo->ipAddr() == 0) {
            foundDaqAliasKey = true;
            break;
          }
        }
      }
    }
    if (foundDaqAliasKey) {
      MsgLog(logger, DBGLVL, "hack to deal with alias from both DAQ and control, found 0 ip address, putting in front");
      EventKey daqAlias = *iter;
      eventKeys.erase(iter);
      eventKeys.push_front(daqAlias);
    } else {
      MsgLog(logger, DBGLVL, "hack to deal with alias from both DAQ and control, did not find 0 ip address");
    }
  }

  // helper class to report on the open identifiers in a file, and close them.
  struct H5OpenObjects {

    std::vector<hid_t> openFile, openDataset, openGroup, openDatatype, openAttr;
    ssize_t FILE, DATASET, GROUP, DATATYPE, ATTR, ALL;

    H5OpenObjects(hdf5pp::File file) {
      FILE = H5Fget_obj_count(file.id(), H5F_OBJ_FILE | H5F_OBJ_LOCAL);
      DATASET = H5Fget_obj_count(file.id(), H5F_OBJ_DATASET | H5F_OBJ_LOCAL);
      GROUP = H5Fget_obj_count(file.id(), H5F_OBJ_GROUP | H5F_OBJ_LOCAL);
      ATTR = H5Fget_obj_count(file.id(), H5F_OBJ_ATTR | H5F_OBJ_LOCAL);
      DATATYPE = H5Fget_obj_count(file.id(), H5F_OBJ_DATATYPE | H5F_OBJ_LOCAL);
      ALL = FILE + DATASET + GROUP + ATTR + DATATYPE;
      setOpen(file.id(), openFile,     FILE,     H5F_OBJ_FILE);
      setOpen(file.id(), openGroup,    GROUP,    H5F_OBJ_GROUP);
      setOpen(file.id(), openDatatype, DATATYPE, H5F_OBJ_DATATYPE);
      setOpen(file.id(), openAttr,     ATTR,     H5F_OBJ_ATTR);
      setOpen(file.id(), openDataset,  DATASET,  H5F_OBJ_DATASET);
    }
    
    std::string dumpStr(bool detailed=false) {
      ostringstream msg;
      msg << "GROUP=" << GROUP
          << " DATASET=" << DATASET
          << " ATTR=" << ATTR
          << " DATATYPE=" << DATATYPE
          << " FILE=" << FILE
          << " ALL=" << ALL;
      if (detailed) {
        msg << std::endl;
        msg << reportOpenNames(openFile, "File");
        msg << reportOpenNames(openGroup, "Group");
        msg << reportOpenNames(openDataset, "Dataset");
        msg << reportOpenNames(openDatatype, "Datatype");
        msg << reportOpenNames(openAttr, "Attr");
      }
      return msg.str();
    }
    
    void closeOpenNonFileIds() {
      closeIds(openGroup, "Group");
      closeIds(openDataset, "Dataset");
      closeIds(openDatatype, "Datatype");
      closeIds(openAttr, "Attr");
    }

    private:

    void closeIds(std::vector<hid_t> &ids, const char *label) {
      for (unsigned idx = 0; idx < ids.size(); ++idx) {
        hid_t id = ids[idx];
        if (H5Iis_valid(id) > 0) {
          herr_t err = 0;
          switch(H5Iget_type(id)) {
          case H5I_FILE:
          case H5I_BADID:
            MsgLog(logger, warning, "H5OpenObjects::closeIds - " << label << " id=" << id << " is file or badid");
            break;
          case H5I_GROUP:
            err = H5Gclose(id);
            break;
          case H5I_DATATYPE:
            err = H5Tclose(id);
            break;
          case H5I_DATASPACE:
            err = H5Sclose(id);
            break;
          case H5I_DATASET:
            err = H5Dclose(id);
            break;
          case H5I_ATTR:
            err = H5Aclose(id);
            break;
          default:
            break;
          } // switch
          if (err<0) MsgLog(logger, error, "H5OpenObjects::closeIds - " << label << " error closing id=" << id);
        } else {
          MsgLog(logger, warning, "H5OpenObjects::closeIds - " << label << " id=" << id << " is not valid");
        } 
      }
    }
    
    std::string reportOpenNames(std::vector<hid_t> &ids, const char *label) {
      const int NAMELEN = 512;
      char name[NAMELEN];
      ostringstream msg;
      msg << "**" << label << "**" << std::endl;
      for (unsigned idx = 0; idx < ids.size(); ++idx) {
        msg << "  id=" << ids[idx] << " name=";
        ssize_t ret = H5Iget_name(ids[idx], name, NAMELEN);
        if (ret < 0) msg << " --error-- bad identifier";
        else if (ret == 0) msg << " -- no name associated with identifier --";
        else msg << name;
        msg << std::endl;
      }
      return msg.str();
    }

    void setOpen(hid_t fileid, std::vector<hid_t> &toFill, ssize_t count, unsigned types) {
      if (count == 0) return;
      toFill.resize(count);
      hid_t *obj_id_list = &(toFill[0]);
      if (obj_id_list == 0) MsgLog(logger, error, "null obj id list?");
      ssize_t filled = H5Fget_obj_ids( fileid, types, count, &toFill[0]);
      if (filled != count) MsgLog(logger, error, "H5OpenObjects::setOpen did not fill expected " 
                                  << count << " for type " << types << " it filled: " << filled);
    }
  }; // H5OpenObjects

  
}; // local namespace

const std::string & H5Output::msgLoggerName() {
  return logger;
}

//             ----------------------------------------
//             -- Public Function Member Definitions --
//             ----------------------------------------

/////////////////////////////////////////
// constructor and initialization methods:

H5Output::H5Output(string moduleName) : Module(moduleName, 
                                               true), // observe all events, even skipped
                                        m_currentConfigureCounter(0),
                                        m_currentRunCounter(0),
                                        m_currentCalibCycleCounter(0),
                                        m_currentEventCounter(0),
                                        m_totalEventsProcessed(0),
                                        m_totalCalibCyclesProcessed(0),
                                        m_maxSavedPreviousSplitEvents(0),
                                        m_totalConfigStoreUpdates(-1),
                                        m_storeEpics(EpicsH5GroupDirectory::Unknown),
                                        m_printedNotFilteringWarning(false)
{
  MsgLog(logger,TRACELVL,name() << " constructor()");
}

void H5Output::init() {
  std::string output_file = readConfigParameters();
  m_splitScanMgr = boost::make_shared<SplitScanMgr>(output_file,
                                                    m_splitCCInSubDir,
                                                    m_split,
                                                    m_mpiWorkerStartCalibCycle);
  if (m_splitScanMgr->thisJobWritesMainOutputFile()) {
    m_h5fileName = output_file;
  } else {
    if (not m_splitScanMgr->isMPIWorker()) MsgLog(logger, fatal, "unexpected - not mpi worker");
    m_h5fileName = m_splitScanMgr->getExtFilePath();
  }
  if (not m_quiet) MsgLog(logger, info, "output file: " << m_h5fileName);

  m_hdfWriters.initialize();
  bool excludePsana = filterHdfWriterMap();
  if (excludePsana) {
    if (m_storeEpics != EpicsH5GroupDirectory::DoNotStoreEpics) {
      MsgLog(logger, info, "setting store_epics to 'no' as type filter is 'exclude psana'");
    }
    m_storeEpics = EpicsH5GroupDirectory::DoNotStoreEpics;
  }
  m_hdfWriterEventId = boost::make_shared<HdfWriterEventId>();
  m_hdfWriterDamage = boost::make_shared<HdfWriterDamage>();
  m_hdfWriterEventId->setDatasetCreationProperties(m_eventIdCreateDsetProp);
  m_hdfWriterDamage->setDatasetCreationProperties(m_damageCreateDsetProp);
  m_epicsGroupDir.initialize(m_storeEpics,
                             m_hdfWriterEventId,
                             m_epicsPvCreateDsetProp,
                             m_defaultCreateDsetProp);
  m_configureGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  m_calibStoreGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  m_calibCycleGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  m_calibCycleEndGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  m_runGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  m_runEndGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  m_configureEndGroupDir.setEventIdAndDamageWriters(m_hdfWriterEventId,m_hdfWriterDamage);
  boost::shared_ptr<HdfWriterDamage> nullHdfWriterDamage;
  TypeAliases::Alias2TypesMap::const_iterator pos = m_typeAliases.alias2TypesMap().find("ndarray_types");
  if (pos == m_typeAliases.alias2TypesMap().end()) MsgLog(logger, fatal, "The TypeAliases map does not include ndarray_types as a key");
  if ((pos->second).size() == 0) MsgLog(logger, fatal, "There are no types assigned to the  'ndarray_types' alias in the TypeAliases map");
  m_h5groupNames = boost::make_shared<H5GroupNames>(m_calibration_key, pos->second);
  m_configureGroupDir.setH5GroupNames(m_h5groupNames);
  m_calibStoreGroupDir.setH5GroupNames(m_h5groupNames);
  m_calibCycleGroupDir.setH5GroupNames(m_h5groupNames);
  m_calibCycleEndGroupDir.setH5GroupNames(m_h5groupNames);
  m_runEndGroupDir.setH5GroupNames(m_h5groupNames);
  m_runGroupDir.setH5GroupNames(m_h5groupNames);
  m_configureEndGroupDir.setH5GroupNames(m_h5groupNames);
  createH5OutputFile();
}

list<string> H5Output::configListReportIfNotDefault(const string &param, 
                                                    const list<string> &defaultList) const {
  list<string> value = configList(param, defaultList);
  if ((not m_quiet) and (value != defaultList)) {
    WithMsgLog(logger,info,stream) {
      stream << "param " << param << " = ";
      std::ostream_iterator<string> out_it (stream," ");
      std::copy ( value.begin(), value.end(), out_it );
      stream << " (not default value)";
    }
  }
  return value;
}

std::string H5Output::readConfigParameters() {
  std::list<std::string> remainingConfigKeys = configSvc().getKeys(name());
  MsgLog(logger, TRACELVL, name() << " reading config parameters");
  m_quiet = config("quiet",false);
  remainingConfigKeys.remove("quiet");
  std::string output_file = configStr("output_file");
  remainingConfigKeys.remove("output_file");

  // default list for type_filter, src_filter, ndarray_key_filter and std_string_key_filter
  list<string> include_all;
  include_all.push_back("include");
  include_all.push_back("all");

  // type filter parameters
  const set<string> & typeAliases = m_typeAliases.aliases();
  set<string>::const_iterator alias;
  for (alias = typeAliases.begin(); alias != typeAliases.end(); ++alias) {
    m_typeInclude[*alias] = excludeIncludeToBool(*alias,configStr(*alias,"include"));
    remainingConfigKeys.remove(*alias);

    if ((not m_quiet) and (not m_typeInclude[*alias])) {
      MsgLog(logger,info,"param " << *alias << " = exclude (not default)");
    }
  }
  m_type_filter = configListReportIfNotDefault("type_filter", include_all);
  remainingConfigKeys.remove("type_filter");

  map<string, EpicsH5GroupDirectory::EpicsStoreMode> validStoreEpicsInput;
  validStoreEpicsInput["no"]=EpicsH5GroupDirectory::DoNotStoreEpics;
  validStoreEpicsInput["calib_repeat"]=EpicsH5GroupDirectory::RepeatEpicsEachCalib;
  validStoreEpicsInput["updates_only"]=EpicsH5GroupDirectory::OnlyStoreEpicsUpdates;
  validStoreEpicsInput["always"]=EpicsH5GroupDirectory::StoreAllEpicsOnEveryShot;
  string storeEpics = configReportIfNotDefault(string("store_epics"), string("calib_repeat"));
  remainingConfigKeys.remove("store_epics");
  
  map<string, EpicsH5GroupDirectory::EpicsStoreMode>::iterator epicsModeUserInput = validStoreEpicsInput.find(storeEpics);
  if (epicsModeUserInput == validStoreEpicsInput.end()) {
    MsgLog(logger, fatal, "config parameter 'epics_store' must be one of 'calib_repeat' 'updates_only' 'always' or 'no'. The value: '"
           << storeEpics << "' is invalid");
  }
  m_storeEpics = epicsModeUserInput->second;

  // check for SplitScan before reporting on epics, we may override it
  string splitStr = configReportIfNotDefault(string("split"),string("NoSplit"));
  remainingConfigKeys.remove("split");
  m_splitCCInSubDir = configReportIfNotDefault(string("split_cc_in_subdir"),false);
  remainingConfigKeys.remove("split_cc_in_subdir");
  if (splitStr == "NoSplit") m_split = SplitScanMgr::NoSplit;
  else if (splitStr == "MPIWorker") m_split = SplitScanMgr::MPIWorker;
  else if (splitStr == "MPIMaster") m_split = SplitScanMgr::MPIMaster;
  else MsgLog(logger,fatal,"config parameter 'split' must be 'NoSplit', 'MPIWorker' or 'MPIMaster' (default is NoSplit)");
  if (m_split == SplitScanMgr::MPIWorker) {
    m_mpiWorkerStartCalibCycle = config("first_calib_cycle_number");
    remainingConfigKeys.remove("first_calib_cycle_number");
    if (m_mpiWorkerStartCalibCycle < 0) {
      MsgLog(logger, fatal, "MPI Worker started with negative calib cycle: " << m_mpiWorkerStartCalibCycle);
    }
  }
  if ((m_split != SplitScanMgr::NoSplit) and 
      (m_storeEpics == EpicsH5GroupDirectory::OnlyStoreEpicsUpdates)) {
    m_storeEpics = EpicsH5GroupDirectory::RepeatEpicsEachCalib;
    MsgLog(logger,warning,"epics storage set to 'calib_repeat' (was 'updates_only') since SplitScan mode in use");
  }
  MsgLog(logger,TRACELVL, name() << "epics storage: " << EpicsH5GroupDirectory::epicsStoreMode2str(m_storeEpics));

  m_overwrite = configReportIfNotDefault("overwrite",false);
  remainingConfigKeys.remove("overwrite");

  // src filter parameters
  m_src_filter = configListReportIfNotDefault("src_filter", include_all);
  remainingConfigKeys.remove("src_filter");
  
  // typesrckey filter parameters
  m_eventkey_filter = configListReportIfNotDefault("eventkey_filter", include_all);
  remainingConfigKeys.remove("eventkey_filter");
  
  m_unknown_src_ok = configReportIfNotDefault("unknown_src_ok",false);
  remainingConfigKeys.remove("unknown_src_ok");

  // key filter parameters
  m_key_filter = configListReportIfNotDefault("key_filter", include_all);
  remainingConfigKeys.remove("key_filter");

  // other translation parameters, calibration, metadata
  m_skip_calibrated = configReportIfNotDefault("skip_calibrated",false);
  remainingConfigKeys.remove("skip_calibrated");
  m_calibration_key = configReportIfNotDefault(string("calibration_key"),string("calibrated"));
  remainingConfigKeys.remove("calibration_key");
  m_exclude_calibstore = configReportIfNotDefault("exclude_calibstore",false);
  remainingConfigKeys.remove("exclude_calibstore");
  if ((m_skip_calibrated) and (not m_exclude_calibstore)) {
    if (not m_quiet) MsgLog(logger,info, "setting skip_calibstore to true since skip_calibrated is true");
    m_exclude_calibstore = true;
  }
  m_minEventsPerMPIWorker = configReportIfNotDefault("min_events_per_calib_file",
						     MIN_EVENTS_PER_CALIB_FILE_DEFAULT);
  remainingConfigKeys.remove("min_events_per_calib_file");

  m_chunkManager.readConfigParameters(*this, remainingConfigKeys);

  m_defaultShuffle = configReportIfNotDefault("shuffle",true);
  m_defaultDeflate = configReportIfNotDefault("deflate",1);
  remainingConfigKeys.remove("shuffle");
  remainingConfigKeys.remove("deflate");

  bool eventIdShuffle = configReportIfNotDefault("eventIdShuffle",m_defaultShuffle);
  remainingConfigKeys.remove("eventIdShuffle");
  int eventIdDeflate = configReportIfNotDefault("eventIdDeflate",m_defaultDeflate);
  remainingConfigKeys.remove("eventIdDeflate");
  bool damageShuffle = configReportIfNotDefault("damageShuffle",false);
  remainingConfigKeys.remove("damageShuffle");
  int damageDeflate = configReportIfNotDefault("damageDeflate",m_defaultDeflate);
  remainingConfigKeys.remove("damageDeflate");

  bool stringShuffle = configReportIfNotDefault("stringShuffle",false);
  remainingConfigKeys.remove("stringShuffle");
  int stringDeflate = configReportIfNotDefault("stringDeflate",-1);
  remainingConfigKeys.remove("stringDeflate");

  bool epicsPvShuffle = configReportIfNotDefault("epicsPvShuffle",false);
  remainingConfigKeys.remove("epicsPvShuffle");
  int epicsPvDeflate = configReportIfNotDefault("epicsPvDeflate",m_defaultDeflate);
  remainingConfigKeys.remove("epicsPvDeflate");

  bool ndarrayShuffle = configReportIfNotDefault("ndarrayShuffle",m_defaultShuffle);
  remainingConfigKeys.remove("ndarrayShuffle");
  int ndarrayDeflate = configReportIfNotDefault("ndarrayDeflate",m_defaultDeflate);
  remainingConfigKeys.remove("ndarrayDeflate");

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

  m_maxSavedPreviousSplitEvents = configReportIfNotDefault("max_saved_split_events", 3000);
  remainingConfigKeys.remove("max_saved_split_events");

  // remove all the keys that the h5-mpi-translate driver uses but H5Output does not:
  remainingConfigKeys.remove("fast_index");
  remainingConfigKeys.remove("fi_mb_half_block");
  remainingConfigKeys.remove("fi_num_blocks");
  remainingConfigKeys.remove("num_events_check_done_calib_file");
  remainingConfigKeys.remove("printenv");
  
  // now warn the user if there are any unexpected keys
  for (std::list<std::string>::iterator pos = remainingConfigKeys.begin();
       pos != remainingConfigKeys.end(); ++pos) {
    MsgLog(logger, warning, "**unexpected configuration key** check spelling. key=" << *pos);
  }
  return output_file;
}

bool H5Output::filterHdfWriterMap() {
  bool isExclude, includeAll;
  set<string> filterSet;
  bool isPsanaExclude = false;
  parseFilterConfigString("type_filter", m_type_filter, isExclude, includeAll, filterSet);
  if (not includeAll) {
    bool hasPsana = false;
    if (filterSet.find("psana") != filterSet.end()) {
      if (filterSet.size() != 1) MsgLog(logger,fatal, "type_filter has 'psana' "
                                 " and other entries. If psana is in type_filter list, it must be the only entry.");
      hasPsana = true;
    }    
    map<string, bool>::iterator typeAliasIter;
    if (hasPsana) {
      isPsanaExclude = isExclude;
      for (typeAliasIter = m_typeInclude.begin(); typeAliasIter != m_typeInclude.end(); ++typeAliasIter) {
        const string &typeAlias = typeAliasIter->first;
        bool &includeFlag = typeAliasIter->second;
        if (typeAlias == "ndarray_types" or typeAlias == "std_string") {
          includeFlag = isExclude;
        } else {
          includeFlag = not isExclude;
        }
      }
    } else {
      for (typeAliasIter = m_typeInclude.begin(); typeAliasIter != m_typeInclude.end(); ++typeAliasIter) {
        typeAliasIter->second = isExclude;
      }
      for (set<string>::iterator filterIter = filterSet.begin(); filterIter != filterSet.end(); ++filterIter) {
        const string &filter = *filterIter;
        typeAliasIter = m_typeInclude.find(filter);
        if (typeAliasIter == m_typeInclude.end()) MsgLog(logger,fatal,"type_filter contains '"<< filter 
                                                         << "' which is an unknown type alias");
        typeAliasIter->second = not isExclude;
      }
    }
  }    
  const TypeAliases::Alias2TypesMap & alias2TypesMap = m_typeAliases.alias2TypesMap();
  TypeAliases::Alias2TypesMap::const_iterator aliasPos;
  for (aliasPos = alias2TypesMap.begin(); aliasPos != alias2TypesMap.end(); ++aliasPos) {
    string alias = aliasPos->first;
    const TypeAliases::TypeInfoSet & typeInfoSet = aliasPos->second;
    if (not m_typeInclude[alias]) {
      removeTypes(m_hdfWriters, typeInfoSet);
    }
  }
  return isPsanaExclude;
}

void H5Output::initializeEventKeyFilter(PSEnv::Env &env) {
  set<string> eventKeyFilterSet;
  parseFilterConfigString("eventkey_filter", m_eventkey_filter,
                          m_eventkeyFilterIsExclude, m_includeAllEventKey, eventKeyFilterSet);
  for (set<string>::iterator pos = eventKeyFilterSet.begin(); pos != eventKeyFilterSet.end(); ++pos) {
    string curArg = *pos;
    size_t firstSepPos = curArg.find("__");
    if (firstSepPos == string::npos) {
      MsgLog(logger, fatal, "No separator, __ found in eventkey_filter arg: " << curArg);
    }
    string typeAlias = curArg.substr(0,firstSepPos);
    string srcStr, keyStr;
    curArg = curArg.substr(firstSepPos+2);
    size_t secondSepPos = curArg.find("__");
    if (secondSepPos == string::npos) {
      srcStr = curArg;
    } else {
      srcStr = curArg.substr(0,secondSepPos);
      keyStr = curArg.substr(secondSepPos+2);
    }
    MsgLog(logger, trace, "eventkeyfilter: type=" << typeAlias << " src=" << srcStr << " key=" << keyStr);
    TypeAliases::Alias2TypesMap alias2types = m_typeAliases.alias2TypesMap();
    if (alias2types.find(typeAlias) == alias2types.end()) {
      MsgLog(logger, fatal, "eventkeyfilter: type=" << typeAlias << " is not known. Use a Translator type alias - see default config file for list.");
    }
    TypeAliases::TypeInfoSet types = alias2types[typeAlias];
    for (TypeAliases::TypeInfoSet::iterator typePos = types.begin(); typePos != types.end(); ++typePos) {
      if (m_eventKeyFilters.find(*typePos) == m_eventKeyFilters.end()) {
        m_eventKeyFilters[*typePos]=SrcKeyList();
      }
      PSEvt::Source source(srcStr);
      try {
        PSEvt::Source::SrcMatch srcMatch = source.srcMatch(*(env.aliasMap()));
        m_eventKeyFilters[*typePos].push_back(std::pair<PSEvt::Source::SrcMatch, std::string>(srcMatch, keyStr));
      } catch (PSEvt::Exception &) {
        if (m_unknown_src_ok) {
          MsgLog(logger, warning, "unknown src " << srcStr << " in eventkey_filter set - ignoring");
        } else {
          MsgLog(logger, fatal, "unknown src " << srcStr << " in eventkey_filter set - check spelling. " << std::endl
                 << " To proceed anyways, add the option 'unknown_src_ok=True' to the Translator configuration");
        }
      }
    }
  }
}

void H5Output::initializeSrcAndKeyFilters(PSEnv::Env &env) {
  set<string> srcNameFilterSet;
  parseFilterConfigString("src_filter", m_src_filter, m_srcFilterIsExclude, m_includeAllSrc, srcNameFilterSet);
  m_psevtSourceFilterList.clear();
  for (set<string>::iterator pos = srcNameFilterSet.begin(); pos != srcNameFilterSet.end(); ++pos) {
    PSEvt::Source source(*pos);
    try {
      PSEvt::Source::SrcMatch srcMatch = source.srcMatch(*(env.aliasMap()));
      m_psevtSourceFilterList.push_back(srcMatch);
    } catch (PSEvt::Exception &) {
      if (m_unknown_src_ok) {
        MsgLog(logger, warning, "unknown src " << *pos << " in src_filter set - ignoring");
      } else {
        MsgLog(logger, fatal, "unknown src " << *pos << " in src_filter set. " << std::endl
               << " To proceed anyways, add the option 'unknown_src_ok=True' to the Translator configuration");
      }
    }
  }
  MsgLog(logger,TRACELVL, name() << "src_filter: isExclude=" << m_srcFilterIsExclude << " all=" << m_includeAllSrc);
  parseFilterConfigString("key_filter", m_key_filter, m_keyFilterIsExclude,   m_includeAllKey,   m_keyFilterSet, true);
  if (m_keyFilterSet.find(doNotTranslatePrefix()) != m_keyFilterSet.end()) {
    MsgLog(logger,warning, "key_filter contains special key string: " 
           << doNotTranslatePrefix() << " it has been removed.");
    m_keyFilterSet.erase(doNotTranslatePrefix());
  }
  if (m_keyFilterSet.find(ndarrayVlenPrefix()) != m_keyFilterSet.end()) {
    MsgLog(logger,warning, "key_filter contains special key string: " 
           << ndarrayVlenPrefix() << " it has been removed.");
    m_keyFilterSet.erase(ndarrayVlenPrefix());
  }
}

void H5Output::createH5OutputFile() {
  m_startTime = LusiTime::Time::now();
  m_translatorTime = 0.0;
  if (not m_quiet) MsgLog(logger,info,name() << " creating h5 output file: " << m_h5fileName);

  unsigned majnum, minnum, relnum;
  herr_t err = H5get_libversion(&majnum, &minnum, &relnum);
  if (err != 0) throw hdf5pp::Hdf5CallException(ERR_LOC,"failed to get Hdf5 library version number");
  MsgLog(logger,TRACELVL,"Hdf Library version info: " << majnum << "." << minnum << "." << relnum);

  hdf5pp::File::CreateMode mode = m_overwrite ? hdf5pp::File::Truncate : hdf5pp::File::Exclusive ;

  // change the size of the B-Tree for chunked datasets
  hdf5pp::PListFileCreate fcpl;
  fcpl.set_istore_k(2);
  fcpl.set_sym_k(2, 2);
  
  // we want to create new file
  hdf5pp::PListFileAccess fapl ;
  if (m_splitScanMgr->isMPIMaster()) {
    // h5py and matlab should be able to read from the master file if we 
    // use CloseWeak. We prefer close strong to make sure all datasets/groups etc are closed.
    // Presently, I believe some h5 objects are not closed properly in psddl_hdf2psana
    fapl.set_fclose_degree(hdf5pp::PListFileAccess::CloseWeak);
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
  if (not m_quiet) MsgLog(logger, info, "Starting translation process " << m_startTime);
  m_h5file.createAttr<const char*> ("origin").store ( "psana-translator" ) ;
  m_h5file.createAttr<const char*> ("created").store ( m_startTime.toString().c_str() ) ;
  m_h5file.createAttr<const char*> ("split_mode").store ( SplitScanMgr::splitModeStr(m_split).c_str());
  if (m_splitScanMgr->isMPIMaster()) {
    // close and reopen the file, this seems to allow multiple readers to work better
    flushOutputFile();
    m_h5file.close();
    m_h5file = hdf5pp::File::open(m_h5fileName, hdf5pp::File::Update, fapl);
    if (not m_h5file.valid()) {
      MsgLog(logger, fatal, "MPI Master unable to reopen h5 master link file");
    }
  }
  MsgLog(logger, TRACELVL, "file created");
}

bool H5Output::checkIfNewTypeHasSameH5GroupNameAsCurrentTypes(const std::type_info * newType) {
  vector<const std::type_info *>  currentTypes;
  for (unsigned vlenTable = 0; vlenTable < 2; ++vlenTable) {
    currentTypes = m_hdfWriters.types(bool(vlenTable));
    string key = vlenTable ? "" : "vlen";
    string newH5TypeGroupName = m_h5groupNames->nameForType(newType,key);
    for (unsigned idx = 0; idx < currentTypes.size(); ++idx) {
      const std::type_info * currentType = currentTypes[idx];
      if (*currentType == *newType) continue;
      string currentName = m_h5groupNames->nameForType(currentType,key);
      if (newH5TypeGroupName == currentName) {
        MsgLog(logger, error, "new type "
               << PSEvt::TypeInfoUtils::typeInfoRealName(newType)
               << " gets same hdf5 group: " 
               << currentName << " as type: " 
               << PSEvt::TypeInfoUtils::typeInfoRealName(currentType)
               << " **it will NOT be registered**");
        return true;
      }
    }
  }
  return false;
}

void H5Output::checkForNewWriters(PSEvt::Event &evt) {
  list<EventKey> eventKeys = evt.keys();
  list<EventKey>::iterator keyIter;
  for (keyIter = eventKeys.begin(); keyIter != eventKeys.end(); ++keyIter) {
    EventKey &eventKey = *keyIter;
    const std::type_info & keyCppType = *eventKey.typeinfo();
    if (keyCppType == typeid(Translator::HdfWriterNew)) {
      const Pds::Src &src = eventKey.src();
      const string &key = eventKey.key();
      boost::shared_ptr<Translator::HdfWriterNew> newWriter = evt.get(src,key);
      if (not newWriter) {
        MsgLog(logger,error," unexpected: could not retrieve new writer");
        return;
      }
      const std::type_info * newType = newWriter->typeInfoPtr();
      bool nameCollision = checkIfNewTypeHasSameH5GroupNameAsCurrentTypes(newType);
      if (nameCollision) continue;
      MsgLog(logger,TRACELVL,name()<<" new hdf5 writer found for type: "
             << PSEvt::TypeInfoUtils::typeInfoRealName(newType)
             << " key: " << key);
      boost::shared_ptr<HdfWriterNewDataFromEvent> newWriterFromEvent = 
        boost::make_shared<HdfWriterNewDataFromEvent>(*newWriter,key);
      // overwrite the old writer if it is there.
      bool replaced = m_hdfWriters.replace(newType, newWriterFromEvent, NewWriterType);
      if (replaced) {
        MsgLog(logger, warning, " overwriting previous writer for type"
               << PSEvt::TypeInfoUtils::typeInfoRealName(newType));
      }
    }
  }
}

//////////////////////////////////////////////////////////
// Event Processing processing - Module methods

void H5Output::beginJob(Event& evt, Env& env) 
{
  MsgLog(logger,DBGLVL,"H5Output beginJob()");
  init();
  boost::shared_ptr<EventId> eventId = evt.get();
  initializeSrcAndKeyFilters(env);
  initializeEventKeyFilter(env);
  m_configureGroupDir.setAliasMap(env.aliasMap());
  m_calibStoreGroupDir.setAliasMap(env.aliasMap());
  m_calibCycleGroupDir.setAliasMap(env.aliasMap());
  m_calibCycleEndGroupDir.setAliasMap(env.aliasMap());
  m_runEndGroupDir.setAliasMap(env.aliasMap());
  m_runGroupDir.setAliasMap(env.aliasMap());
  m_configureEndGroupDir.setAliasMap(env.aliasMap());

  m_h5file.createAttr<uint32_t> ("expNum").store ( env.expNum() ) ;
  m_h5file.createAttr<const char*> ("experiment").store ( env.experiment().c_str() ) ;
  m_h5file.createAttr<const char*> ("instrument").store ( env.instrument().c_str() ) ;
  m_h5file.createAttr<const char*> ("jobName").store ( env.jobName().c_str() ) ;

  m_currentRunCounter = 0;
  checkForNewWriters(evt);
  createNextConfigureGroup(eventId);
  m_configureGroupDir.clearMaps();
  m_chunkManager.beginJob(env);
  
  addConfigTypes(evt, env, m_configureGroupDir, m_currentConfigureGroup);
  m_epicsGroupDir.processBeginJob(m_currentConfigureGroup.id(), 
                                  env.epicsStore(), eventId);
}

void H5Output::beginRun(Event& evt, Env& env) 
{
  MsgLog(logger,DBGLVL, name() << ": beginRun()");
  boost::shared_ptr<EventId> eventId = evt.get();
  // the aliasMap can change from run to run, so reinitialize the src filter list with each run.
  initializeSrcAndKeyFilters(env);
  if (m_splitScanMgr->isMPIWorker()) {    
    m_currentCalibCycleCounter =  m_mpiWorkerStartCalibCycle;
  } else {
    m_currentCalibCycleCounter = 0;
  }
  createNextRunGroup(eventId);
  m_calibratedEventKeys.clear();

  if ((m_currentRunCounter > 0) and (m_splitScanMgr->splitScanMode())) {
    MsgLog(logger,error,"cannot process multiple runs in split scan mode"
	   << " - external calib filenames will not be unique - stopping early.");
    stop();
  }

  addConfigTypes(evt, env, m_runGroupDir, m_currentRunGroup);

}

void H5Output::beginCalibCycle(Event& evt, Env& env) 
{
  MsgLog(logger,DBGLVL,"beginCalibCycle() calib cycle " << m_currentCalibCycleCounter);
  if (m_splitScanMgr->isMPIMaster()) {
    MsgLog(logger, fatal, "beginCalibCycle should not be called from the mpi master");
  }
  boost::shared_ptr<EventId> eventId = evt.get();
  createNextCalibCycleGroup(eventId);
  m_calibCycleGroupDir.clearMaps();
  m_currentEventCounter = 0;
  m_chunkManager.beginCalibCycle(env);
  addConfigTypes(evt, env, m_calibCycleGroupDir, m_currentCalibCycleGroup);
  m_epicsGroupDir.processBeginCalibCycle(m_currentCalibCycleGroup.id(), env.epicsStore());
}

void H5Output::event(Event& evt, Env& env) 
{
  LusiTime::Time startTime = LusiTime::Time::now();
  ++m_totalEventsProcessed;
  if (not psanaSkipedEvent(evt)) {
    try {
      eventImpl(evt, env);
    } catch (...) {
      MsgLog(logger,error,name() << " event: error, closing file: " << m_h5fileName 
             << " calibCycle=" << m_currentCalibCycleCounter
             << " evts (this cc)=" << m_currentEventCounter << " evts(total)=" << m_totalEventsProcessed);
      closeH5FileDueToEventException();
      throw;
    }
  }
  ++m_currentEventCounter;
  LusiTime::Time endTime = LusiTime::Time::now();
  double processingTime = (endTime.sec()-startTime.sec()) + (endTime.nsec()-startTime.nsec())/1e9;
  m_translatorTime += processingTime;
}


bool H5Output::isNDArray( const type_info *typeInfoPtr) {
  return m_h5groupNames->isNDArray(typeInfoPtr);
}

Pds::Damage H5Output::getDamageForEventKey(const EventKey &eventKey,
					   boost::shared_ptr<PSEvt::DamageMap> damageMap) {
  Pds::Damage damage(0);
  if (damageMap) {
    PSEvt::DamageMap::iterator damagePos = damageMap->find(eventKey);
    if (damagePos != damageMap->end()) damage = damagePos->second.value();
  }
  return damage;
}

/// see's if eventKey has been filtered through the type, source, key, or eventkey filters.
/// optionally checks if it should be skipped in lieu of a calibrated key.
//  The calibration check should be done for event data, but not for config
//  data.  If checkForCalibratedKey is true, a calibrated key is looked for in evt.
//  Returns a null HdfWriter if:
//             there is no writer for this type
//             the src for the eventKey is filtered and it has no key string
//             the type for the eventKey is filtered
//             the key is for std::string or ndarray and the key for this EventKey is filtered
//             the full event key is filtered
//             the type is one of the special calibrated types, a calibrated version 
//               of the key exists, and we are not storing uncalibrated data

boost::shared_ptr<HdfWriterFromEvent> H5Output::checkTranslationFilters(PSEvt::Event &evt,
                                                                        const EventKey &eventKey, 
                                                                        bool checkForCalibratedKey,
                                                                        TypeClass &typeClass)
{
  typeClass = UnknownType;
  const type_info * typeInfoPtr = eventKey.typeinfo();
  const Pds::Src & src = eventKey.src();
  const string & key = eventKey.key();
  boost::shared_ptr<HdfWriterFromEvent> nullHdfWriter;
  if (srcIsFiltered(src)) {
    if (key.size() == 0)  {
      MsgLog(logger,DBGLVL,"Filtering " << eventKey << " due to src_filter");
      return nullHdfWriter;
    } else {
      MsgLog(logger,DBGLVL,"Although src_filter applies to " << eventKey << " it has a non-empty key string. NOT filtering");
    }
  }
  bool hasVlenPrefix = false;
  string stripPrefixKey;
  if (key.size() != 0) {
    hasVlenPrefix = hasNDArrayVlenPrefix(key, &stripPrefixKey);
    if (keyIsFiltered(stripPrefixKey)) {
      MsgLog(logger,DBGLVL,"key is filtered for " << eventKey << ", filtering.");
      return nullHdfWriter;
    }
  }
  if (fullEventKeyIsFiltered(typeInfoPtr, src, key)) {
    MsgLog(logger,DBGLVL,"Filtering " << eventKey << " due to eventkey_filter");
    return nullHdfWriter;
  }
  if (m_skip_calibrated and (stripPrefixKey == m_calibration_key)) {
    if ((hasVlenPrefix) and (not m_h5groupNames->isNDArray(typeInfoPtr))) {
      MsgLog(logger,warning,"eventKey " << eventKey 
             << " contains key with special prefix only for ndarrays, "
             << " but type is not a known ndarray");
    }
    MsgLog(logger,DBGLVL,"skipping calibrated data: " << eventKey);
    return nullHdfWriter;
  }

  typeClass = NdarrayType;
  boost::shared_ptr<HdfWriterFromEvent> hdfWriter = hasVlenPrefix ? m_hdfWriters.findVlen(typeInfoPtr) : m_hdfWriters.find(typeInfoPtr, &typeClass);
  if (not hdfWriter) {
    if (hasVlenPrefix and (not m_h5groupNames->isNDArray(typeInfoPtr))) {
      // The vlen prefix can only be added for ndarrays. Adding it to other types is not
      // supported. However if we can detect that the user did this, print a warning.
      // If ndarrays weren't filtered, then print the warning. Check for an ndarray 
      // (any ndarray will do to see if they are filtered or not)
      bool ndArraysNotFiltered = m_hdfWriters.find(&typeid(ndarray<uint8_t,1>));
      if (ndArraysNotFiltered) {
        MsgLog(logger, warning, "vlen prefix found for key: " 
               << eventKey << " but the type is not a known ndarray");
      }
    }
    MsgLog(logger,DBGLVL,"No hdfwriter found for " << eventKey);
    return nullHdfWriter;
  }
  if ((not m_skip_calibrated) and checkForCalibratedKey) {
    if (key.size()==0) {
      EventKey calibratedKey(typeInfoPtr,src, m_calibration_key);
      if (evt.proxyDict()->exists(calibratedKey)) {
        MsgLog(logger, DBGLVL, "calibrated key exists for " << eventKey << " filtering");
        return nullHdfWriter;
      }
    }
  }
  return hdfWriter;
}

list<EventKey> H5Output::getUpdatedConfigKeys(PSEnv::Env &env) {
  list<EventKey> updatedConfigKeys;
  const PSEvt::HistI * configHist  = env.configStore().proxyDict()->hist();
  if (not configHist) MsgLog(logger,fatal,"getUpdatedConfigKeys - Internal error - no HistI object in configStore");
  if (m_totalConfigStoreUpdates > -1 and (configHist->totalUpdates() <= m_totalConfigStoreUpdates)) {
    return updatedConfigKeys;
  }
  m_totalConfigStoreUpdates = configHist->totalUpdates();
  list<EventKey> configKeys = env.configStore().keys();
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

void H5Output::setEventKeysToTranslate(PSEvt::Event &evt, PSEnv::Env &env,
                                       list<EventKeyTranslation> & toTranslate, 
                                       bool & eventIsFiltered) {
  toTranslate.clear();
  eventIsFiltered = false;
  boost::shared_ptr<PSEvt::DamageMap> damageMap = evt.get();
  list<EventKey> eventKeysFromEvent = evt.keys();
  WithMsgLog(logger, DBGLVL, str) {
    str << "setEventKeysToTranslate - eventKeysFromEvent:";
    for (list<EventKey>::iterator pos = eventKeysFromEvent.begin();
         pos != eventKeysFromEvent.end(); ++pos) {
      str << " " << *pos;
    }
  }
  list<EventKey> updatedConfigKeys = getUpdatedConfigKeys(env);
  WithMsgLog(logger,TRACELVL,str) {
    str << "setEventKeysToTranslate updated config keys:";
    for (list<EventKey>::iterator pos = updatedConfigKeys.begin();
         pos != updatedConfigKeys.end(); ++pos) {
      str << " " << *pos;
    }
  }
  set<EventKey> nonBlanks;
  list<EventKey>::iterator keyIter;
  for (keyIter = updatedConfigKeys.begin(); keyIter != updatedConfigKeys.end(); ++keyIter) {
    TypeClass typeClass;
    boost::shared_ptr<HdfWriterFromEvent> hdfWriter = checkTranslationFilters(evt,*keyIter,false, typeClass);
    if (not hdfWriter) continue;
    Pds::Damage damage = getDamageForEventKey(*keyIter, damageMap);
    toTranslate.push_back(EventKeyTranslation(*keyIter, damage, hdfWriter,
                                              EventKeyTranslation::NonBlank, 
                                              inConfigStore,
                                              typeClass));
    nonBlanks.insert(*keyIter);
  }

  list<EventKeyTranslation> toTranslateFromEvent;
  for (keyIter = eventKeysFromEvent.begin(); keyIter != eventKeysFromEvent.end(); ++keyIter) {
    bool eventFilterKey = hasDoNotTranslatePrefix(keyIter->key());
    if (eventFilterKey) {
      eventIsFiltered = true;
      continue;
    }

    if (eventIsFiltered) continue;

    TypeClass typeClass;
    boost::shared_ptr<HdfWriterFromEvent> hdfWriter = checkTranslationFilters(evt, *keyIter,true, typeClass);
    if (not hdfWriter) continue;

    Pds::Damage damage = getDamageForEventKey(*keyIter, damageMap);
    toTranslateFromEvent.push_back(EventKeyTranslation(*keyIter, damage, hdfWriter,
                                                       EventKeyTranslation::NonBlank, 
                                                       inEvent,
                                                       typeClass));
    nonBlanks.insert(*keyIter);
  }

  // if the event is filtered, still return updated config keys, but nothing from the
  // event for translation to the CalibCycle group
  if (eventIsFiltered) {
    MsgLog(logger,TRACELVL,"setEventKeysToTranslate - event is filtered.");
    WithMsgLog(logger,TRACELVL,str) {
      str << "setEventKeysToTranslate - EventKeyTranslation list: ";
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
  if (damageMap) {
    PSEvt::DamageMap::iterator damagePos;
    for (damagePos = damageMap->begin(); damagePos != damageMap->end(); ++damagePos) {
      Pds::Damage damage = damagePos->second;
      if (damage.value()==0) continue;
      const EventKey &eventKey = damagePos->first;
      bool alreadyAddedAsNonBlank = nonBlanks.find(eventKey) != nonBlanks.end();
      if (alreadyAddedAsNonBlank) continue;

      TypeClass typeClass;
      boost::shared_ptr<HdfWriterFromEvent> hdfWriter = checkTranslationFilters(evt, eventKey, true, typeClass);
      if (not hdfWriter) continue;

      toTranslate.push_back(EventKeyTranslation(eventKey,damage,hdfWriter,
                                                EventKeyTranslation::Blank,
                                                inEvent,
                                                typeClass));
    }
  }

  WithMsgLog(logger,TRACELVL,str) {
    str << " EventKeyTranslation list: ";
    list<EventKeyTranslation>::iterator pos;
    for (pos = toTranslate.begin(); pos != toTranslate.end(); ++pos) {
      EventKeyTranslation & eventKeyTranslation = *pos;
      str << eventKeyTranslation;
      str << ", ";
    }
  }
}


void H5Output::eventImpl(PSEvt::Event &evt, PSEnv::Env &env) 
{
  boost::shared_ptr<EventId> eventId = evt.get();
  boost::shared_ptr<DamageMap> damageMap = evt.get();
  m_epicsGroupDir.processEvent(env.epicsStore(), eventId);
  list<EventKeyTranslation> toTranslate;
  bool eventIsFiltered;
  setEventKeysToTranslate(evt, env, toTranslate, eventIsFiltered);
  vector<pair<Pds::Src,Pds::Damage> > droppedContribs = damageMap->getSrcDroppedContributions();
  bool splitEvent = droppedContribs.size()>0;
  bool repeatEvent = false;
  map<EventKey, long> previousBlanksForRepeatSplitEvent;
  set<EventKey> previousNonBlanksForRepeatSplitEvent;
  if (splitEvent) {
    MsgLog(logger,TRACELVL,"split event " << eventPosition());
    BlankNonBlanksMap::iterator pos = m_previousSplitEvents.find(eventId);
    repeatEvent = pos != m_previousSplitEvents.end();
    if (repeatEvent) {
      MsgLog(logger,TRACELVL,"split event AND repeat event"); 
      previousBlanksForRepeatSplitEvent = pos->second.blanks;
      previousNonBlanksForRepeatSplitEvent = pos->second.nonblanks;
    }
  }

  m_calibCycleGroupDir.markAllSrcKeyGroupsNotWrittenForEvent();
  MsgLog(logger, DBGLVL, name() << eventPosition() << " eventId: " << *eventId);
  list<EventKeyTranslation>::iterator keyIter;
  for (keyIter = toTranslate.begin(); keyIter != toTranslate.end(); ++keyIter) {
    const EventKey eventKey = keyIter->eventKey;
    bool writeBlank = keyIter->entryType == EventKeyTranslation::Blank;
    DataTypeLoc dataTypeLoc = keyIter->dataTypeLoc;
    Pds::Damage damage = keyIter->damage;
    boost::shared_ptr<HdfWriterFromEvent> hdfWriter = keyIter->hdfWriter;
    TypeClass typeClass = keyIter->typeClass;
    const Pds::Src & src = eventKey.src();
    MsgLog(logger,DBGLVL,"eventKey=" << eventKey << "damage= " << damage.value() <<
           " writeBlank=" << writeBlank << " loc=" << dataTypeLoc << " hdfwriter=" << hdfWriter);
    const std::string  & key = eventKey.key();
    if (key == m_calibration_key) {
      m_calibratedEventKeys.insert(eventKey);
      MsgLog(logger,DBGLVL, "eventKey is calibration key. Adding Pds::Src to calibratedSrc Src.");
    }
    TypeMapContainer::iterator typePos = m_calibCycleGroupDir.findType(eventKey);
    if (typePos == m_calibCycleGroupDir.endType()) {
      m_calibCycleGroupDir.addTypeGroup(eventKey, m_currentCalibCycleGroup);
      MsgLog(logger,TRACELVL, "eventKey: " << eventKey 
             << " with group name " << m_h5groupNames->nameForType(eventKey.typeinfo(), eventKey.key())
             << " not in calibCycleEventGroupDir.  Added type to groups");
    }
    SrcKeyMap::iterator srcKeyPos = m_calibCycleGroupDir.findSrcKey(eventKey);
    if (srcKeyPos == m_calibCycleGroupDir.endSrcKey(eventKey)) {
      MsgLog(logger,TRACELVL,
             "src " << src << " not in type group.  Adding src to type group");
      SrcKeyGroup & srcKeyGroup = m_calibCycleGroupDir.addSrcKeyGroup(eventKey,hdfWriter);
      if (writeBlank) {
        MsgLog(logger,TRACELVL," initial event is blank.  Only creating time/damage datasets");
        srcKeyGroup.make_timeDamageDatasets();
      } else {
        MsgLog(logger,TRACELVL," initial event is nonblank.  Creating event datasets, and time/damage datasets");
        srcKeyGroup.make_datasets(dataTypeLoc, evt, env, lookUpDataSetCreationProp(typeClass));
      }
      srcKeyPos = m_calibCycleGroupDir.findSrcKey(eventKey);
    }
    SrcKeyGroup & srcKeyGroup = srcKeyPos->second;
    bool needToAppend = true;
    if (repeatEvent) {
      map<PSEvt::EventKey, long>::iterator previousBlank;
      previousBlank = previousBlanksForRepeatSplitEvent.find(eventKey);
      if (previousBlank != previousBlanksForRepeatSplitEvent.end()) {
        MsgLog(logger,TRACELVL,
               "repeat event has blank entry for eventKey " << eventKey);
        size_t blankIndex = previousBlank->second;
        if (writeBlank) srcKeyGroup.overwriteDamage(blankIndex, eventKey, eventId, damage);
        else srcKeyGroup.overwriteDataAndDamage(blankIndex, eventKey, dataTypeLoc, evt, env, damage);
        srcKeyGroup.written(true);
        needToAppend = false;
      } else {
        MsgLog(logger,TRACELVL,
               "repeat event has nonblank entry for eventKey " << eventKey);
        set<PSEvt::EventKey>::iterator previousNonBlank;
        previousNonBlank = previousNonBlanksForRepeatSplitEvent.find(eventKey);
        if (previousNonBlank != previousNonBlanksForRepeatSplitEvent.end()) {
          needToAppend = false;
          // we do not expect this to happend, print warning
          WithMsgLog(logger,warning,str) {
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
        MsgLog(logger,DBGLVL,"appending blank");
        try {
          srcKeyGroup.appendBlankTimeAndDamage(eventKey, eventId, damage);
        } catch (const std::runtime_error &exp) {
          MsgLog(logger, error, "Failed to append blank - second case");
          throw exp;
        }
      }
      else {
        if (not srcKeyGroup.arrayTypeDatasetsCreated()) {
          MsgLog(logger,TRACELVL, "appending nonblank and type dataset not created, making type datasets");
          srcKeyGroup.make_typeDatasets(dataTypeLoc, evt, env, lookUpDataSetCreationProp(typeClass));
        } else {
          MsgLog(logger,DBGLVL,"appending nonblank");
        }
        srcKeyGroup.appendDataTimeAndDamage(eventKey, dataTypeLoc, evt, env, eventId, damage);
      }
      srcKeyGroup.written(true);
    }
  }
  
  if (splitEvent and not repeatEvent) {
    // As a given source can have several types of data, there may be two or more 
    // droppedContributions from a given source.  That is there may be Src repeats in the 
    // droppedContribs list below.  If there is at least one dropped contribution from a
    // source, we will append blanks to all the types that are event based from that source 
    // that were not already written. We qualify on event based in case config data from the
    // calib cycle exists for the source as well.
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
    m_calibCycleGroupDir.getNotWrittenSrcPartition(droppedSrcs,
                                                   droppedSrcsNotWritten,
                                                   otherNotWritten,
                                                   writtenKeys);
    set<Pds::Src>::iterator droppedSrc;
    for (droppedSrc = droppedSrcs.begin(); droppedSrc != droppedSrcs.end(); ++droppedSrc) {
      const Pds::Src & src = *droppedSrc;
      map<Pds::Src, vector<PSEvt::EventKey> >::iterator notWrittenIter;
      notWrittenIter= droppedSrcsNotWritten.find(src);
      if (notWrittenIter == droppedSrcsNotWritten.end()) {
        MsgLog(logger,warning, 
               "dropped src: " << src << " has not been seen before.  Not writing a blank.");
        continue;
      }
      vector<PSEvt::EventKey> & notWritten = notWrittenIter->second;
      for (size_t notWrittenIdx = 0; notWrittenIdx < notWritten.size(); ++notWrittenIdx) {
        PSEvt::EventKey & eventKey = notWritten[notWrittenIdx];
        SrcKeyMap::iterator srcKeyPos = m_calibCycleGroupDir.findSrcKey(eventKey);
        SrcKeyGroup & srcKeyGroup = srcKeyPos->second;
        Pds::Damage damage = src2damage[src];
        if (srcKeyGroup.arrayDatasetsCreated()) {
          try {
            long pos = srcKeyGroup.appendBlankTimeAndDamage(eventKey, eventId, damage);
            blankNonBlanks.blanks[eventKey] = pos;
          } catch (const std::runtime_error &exp) {
            MsgLog(logger, error, exp.what());
            MsgLog(logger, error, "Failed to append blank. notWritten loop. notWrittenIdx=" << notWrittenIdx);
            throw exp;
          }
        }
      }
    }
    for (size_t writtenIdx = 0; writtenIdx < writtenKeys.size(); ++writtenIdx)  {
      PSEvt::EventKey eventKey = writtenKeys[writtenIdx];
      blankNonBlanks.nonblanks.insert(eventKey); 
    }
    if (m_previousSplitEvents.size() < m_maxSavedPreviousSplitEvents) {
      m_previousSplitEvents[eventId]=blankNonBlanks;
    } else {
      WithMsgLog(logger,warning,str) {
        str << "Maximum number of cached splitEvents reached. ";
        str << "Will not be able to fill in blanks for this event: " << *eventId;
      }
    }
  }
}

void H5Output::addConfigTypes(PSEvt::Event &evt, PSEnv::Env &env, 
			      TypeSrcKeyH5GroupDirectory &configGroupDirectory,
                              hdf5pp::Group & parentGroup) {
  boost::shared_ptr<EventId> eventId = evt.get();
  boost::shared_ptr<PSEvt::DamageMap> damageMap = evt.get();
  list<EventKey> envEventKeys = getUpdatedConfigKeys(env);
  problemWithTranslatingBothDAQ_and_Control_AliasLists(envEventKeys);
  list<EventKey> evtEventKeys = evt.keys();
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
      MsgLog(logger,DBGLVL,"addConfigureTypes eventKey: " << *iter << " loc: " << dataLoc);
      TypeClass typeClass;
      boost::shared_ptr<HdfWriterFromEvent> hdfWriter = checkTranslationFilters(evt, eventKey, checkForCalib, typeClass);
      if (not hdfWriter) continue;
      const std::type_info * typeInfoPtr = eventKey.typeinfo();
      const Pds::Src & src = eventKey.src();
      TypeMapContainer::iterator typePos = configGroupDirectory.findType(eventKey);
      if (typePos == configGroupDirectory.endType()) {
        ++newTypes;
        configGroupDirectory.addTypeGroup(eventKey, parentGroup);
        MsgLog(logger,TRACELVL, "eventKey: " << eventKey
               << " with group name " << m_h5groupNames->nameForType(eventKey.typeinfo(), eventKey.key())
               << " not in configGroupDir.  Added type to groups");
        MsgLog(logger,TRACELVL, 
               PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr) <<" not in groups.  Added type to groups");
      }
      SrcKeyMap::iterator srcKeyPos = configGroupDirectory.findSrcKey(eventKey);
      bool alreadyExists = true;
      if (srcKeyPos == configGroupDirectory.endSrcKey(eventKey)) {
        ++newSrcs;
        configGroupDirectory.addSrcKeyGroup(eventKey,hdfWriter);
        MsgLog(logger, TRACELVL,
               " src=" << src << " key='" << eventKey.key() << "' not in type group.  Added src to type group");
        srcKeyPos = configGroupDirectory.findSrcKey(eventKey);
        MsgLog(logger, TRACELVL, "srcKeyPos reset after finding, test if is == end: " 
               << (srcKeyPos == configGroupDirectory.endSrcKey(eventKey)));
        alreadyExists = false;
      }
      SrcKeyGroup & srcKeyGroup = srcKeyPos->second;
      if (dataLoc == inConfigStore) {
        if (alreadyExists) {
          MsgLog(logger,warning,
                 " multiple keys map to same group during addConfigTypes. skipping key: "
                 << eventKey);
        } else {
          try {
            srcKeyGroup.storeData(eventKey, inConfigStore, evt, env);
          } catch (ErrSvc::Issue &issue) {
            configGroupDirectory.dump();
            throw issue;
          }
        }
      } else if (dataLoc == inEvent) {
        if (not eventId) {
          MsgLog(logger, warning, "not translating eventkey: " << eventKey 
                 << " eventId=0. Cannot write timestamp. This can occur"
                 << " when modules end early, such as from mpi split scan");
        } else {
          srcKeyGroup.make_datasets(inEvent, evt, env, lookUpDataSetCreationProp(typeClass));
          Pds::Damage damage = getDamageForEventKey(eventKey, damageMap);
          srcKeyGroup.appendDataTimeAndDamage(eventKey, inEvent, evt, env, eventId, damage);
        }
      }
      ++newDatasets;
    }
  }
  MsgLog(logger, TRACELVL, "created " << newTypes << " new types, " << newSrcs << " new srcs, and "
         << newDatasets << " newDatasets");
}

void H5Output::endCalibCycle(Event& evt, Env& env) {
  boost::shared_ptr<EventId> eventId = evt.get();
  if (not m_splitScanMgr->isMPIMaster()) {
    lookForAndStoreEndData(evt, env, m_currentCalibCycleGroup, 
                           m_calibCycleEndGroupDir, m_currentCalibCycleEndGroup);
  }
  ++m_totalCalibCyclesProcessed;
  m_epicsGroupDir.processEndCalibCycle();
  m_calibCycleGroupDir.closeGroups();
  if (eventId) ::storeClock ( m_currentCalibCycleGroup, eventId->time(), "end" ) ;
  m_currentCalibCycleGroup.close();

  if (m_splitScanMgr->isMPIWorker() and (m_totalEventsProcessed >= m_minEventsPerMPIWorker)) {
    stop();
  } 

  m_chunkManager.endCalibCycle(m_currentEventCounter);
  ++m_currentCalibCycleCounter;
}

void H5Output::endRun(Event& evt, Env& env) 
{
  boost::shared_ptr<EventId> eventId = evt.get();
  if (not m_splitScanMgr->isMPIMaster()) {
    lookForAndStoreEndData(evt, env, m_currentRunGroup, 
                           m_runEndGroupDir, m_currentRunEndGroup);
  } 
  if (eventId) ::storeClock ( m_currentRunGroup, eventId->time(), "end" ) ;
  m_runGroupDir.closeGroups();
  m_currentRunGroup.close();
  ++m_currentRunCounter;
}

void H5Output::addCalibStoreHdfWriters(HdfWriterMap &hdfWriters) {
  vector< boost::shared_ptr<HdfWriterNew> > calibStoreWriters;
  getHdfWritersForCalibStore(calibStoreWriters);
  MsgLog(logger,TRACELVL,"Adding calibstore hdfwriters: got " 
         << calibStoreWriters.size() << " writers");
  vector< boost::shared_ptr<HdfWriterNew> >::iterator iter;
  for (iter = calibStoreWriters.begin(); iter != calibStoreWriters.end(); ++iter) {
    boost::shared_ptr<HdfWriterNew> calibWriterNew = *iter;
    const std::type_info *calibType = calibWriterNew->typeInfoPtr();
    boost::shared_ptr<HdfWriterNewDataFromEvent> calibWriter;
    calibWriter = boost::make_shared<HdfWriterNewDataFromEvent>(*calibWriterNew,"calib-store");
    bool replacedType = hdfWriters.replace(calibType, calibWriter, CalibStoreType);
    if (replacedType) {
      MsgLog(logger, warning, "calib type " << PSEvt::TypeInfoUtils::typeInfoRealName(calibType)
             << " already has writer, OVERWRITING");
    }
  }
}

void H5Output::removeCalibStoreHdfWriters(HdfWriterMap &hdfWriters) {
  vector< boost::shared_ptr<HdfWriterNew> > calibStoreWriters;
  getHdfWritersForCalibStore(calibStoreWriters);
  MsgLog(logger,TRACELVL,"Removing calibstore hdfwriters: got " 
         << calibStoreWriters.size() << " writers");
  vector< boost::shared_ptr<HdfWriterNew> >::iterator iter;
  for (iter = calibStoreWriters.begin(); iter != calibStoreWriters.end(); ++iter) {
    boost::shared_ptr<HdfWriterNew> calibWriter = *iter;
    const std::type_info *calibType = calibWriter->typeInfoPtr();
    bool removed = hdfWriters.remove(calibType);
    if (not removed) {
      MsgLog(logger, warning, "Removed type from calibStore writeres, but type was "
             << " not present. Type = " 
             << PSEvt::TypeInfoUtils::typeInfoRealName(calibType));
    }
  }
}

void H5Output::lookForAndStoreEndData(PSEvt::Event &evt, PSEnv::Env &env, hdf5pp::Group &parentGroup, 
                                      TypeSrcKeyH5GroupDirectory & endDataGroupDir, hdf5pp::Group &endDataGroup) {
  // right now I don't want to create EndData unless there is something to put in it. My check is 
  // currently for anything new in the configStore, however this is not good enough, new things may be 
  // filtered, and there may be new things in the event
  const std::string groupName = "EndData";
  MsgLog(logger, TRACELVL, "lookForAndStoreEndData");
  const PSEvt::HistI * configHist  = env.configStore().proxyDict()->hist();
  if (not configHist) {
    MsgLog(logger,error,"no HistI object in configStore, " << groupName << " will not be created");
    return;
  }
  bool newConfigStoreKeys = (configHist->totalUpdates() >  m_totalConfigStoreUpdates);
  if (not newConfigStoreKeys) {
    MsgLog(logger, TRACELVL, "lookForAndStoreEndData found no new keys in configstore. Note checking event. exiting");
    return;
  } 
  
  if (parentGroup.hasChild(groupName)) {
    MsgLog(logger, error, "unexpected: group " << parentGroup.name() << " already has child " << groupName);
    return;
  }
  endDataGroup = parentGroup.createGroup(groupName);
  endDataGroupDir.clearMaps();
  addConfigTypes(evt, env, endDataGroupDir, endDataGroup);
  MsgLog(logger, DBGLVL, "finished config types");
  endDataGroupDir.closeGroups();
  endDataGroup.close();
}

void H5Output::lookForAndStoreCalibData(PSEvt::Event &evt, PSEnv::Env &env, hdf5pp::Group &parentGroup) {
  addCalibStoreHdfWriters(m_hdfWriters);
  Type2CalibTypesMap type2calibTypeMap;
  getType2CalibTypesMap(type2calibTypeMap);
  bool calibGroupCreated = false;
  hdf5pp::Group calibGroup;
  map<SrcKeyPair, set<const type_info *> , LessSrcKeyPair> 
    calibStoreTypesAlreadyWritten(LessSrcKeyPair(m_h5groupNames->calibratedKey()));
  map<SrcKeyPair, set<const type_info *> , LessSrcKeyPair>::iterator alreadyWrittenIter;
  std::set<PSEvt::EventKey>::iterator calibEventKeyIter;
  MsgLog(logger,DBGLVL, "going to go through " << m_calibratedEventKeys.size() << " calibrated event keys");
  for (calibEventKeyIter = m_calibratedEventKeys.begin(); 
       calibEventKeyIter != m_calibratedEventKeys.end(); ++calibEventKeyIter) {
    const PSEvt::EventKey &calibEventKey = *calibEventKeyIter;
    MsgLog(logger,DBGLVL, "calib key: " << calibEventKey);
    const type_info * calibEventType = calibEventKey.typeinfo();
    const Pds::Src &src = calibEventKey.src();
    vector<const type_info *> calibStoreTypes = type2calibTypeMap[calibEventType];
    MsgLog(logger, DBGLVL, "going to go through " << calibStoreTypes.size() << " types for key");
    for (unsigned idx = 0; idx < calibStoreTypes.size(); ++idx) {
      const type_info * calibStoreType = calibStoreTypes[idx];
      boost::shared_ptr<void> calibStoreData;
      try {
        calibStoreData = 
          env.calibStore().proxyDict()->get(calibStoreType,PSEvt::Source(src),"",NULL);
      } catch (const std::runtime_error &except) {
        MsgLog(logger,error,"Error retreiving data for " 
               << PSEvt::TypeInfoUtils::typeInfoRealName(calibStoreType)
               << " error is: " << except.what() 
               << " skipping and going to next calibStore entry");
        continue;
      }
      if (calibStoreData) {
        MsgLog(logger, DBGLVL, " got calib store data for type " 
               << PSEvt::TypeInfoUtils::typeInfoRealName(calibStoreType) 
               << " for " << calibEventKey);
        boost::shared_ptr<HdfWriterFromEvent> hdfWriter = m_hdfWriters.find(calibStoreType);
        if (not hdfWriter) {
          MsgLog(logger,TRACELVL,"No HdfWriter found for calibStore type: " 
                 << PSEvt::TypeInfoUtils::typeInfoRealName(calibStoreType));
          continue;
        }
        PSEvt::EventKey calibStoreEventKey(calibStoreType, src, "");
        MsgLog(logger,TRACELVL,"calib data and hdfwriter obtained for " << calibStoreEventKey);
        if (not calibGroupCreated) {
          MsgLog(logger,TRACELVL,"Creating CalibStore group");
	  calibGroup = parentGroup.createGroup(calibStoreGroupName);
          calibGroupCreated = true;
        }
        TypeMapContainer::iterator typePos = m_calibStoreGroupDir.findType(calibStoreEventKey);
        if (typePos == m_calibStoreGroupDir.endType()) {
          m_calibStoreGroupDir.addTypeGroup(calibStoreEventKey, calibGroup);
        }
        SrcKeyMap::iterator srcKeyPos = m_calibStoreGroupDir.findSrcKey(calibStoreEventKey);
        if (srcKeyPos == m_calibStoreGroupDir.endSrcKey(calibStoreEventKey)) {
          m_calibStoreGroupDir.addSrcKeyGroup(calibStoreEventKey,hdfWriter);
          srcKeyPos = m_calibStoreGroupDir.findSrcKey(calibStoreEventKey);
        }
        SrcKeyGroup & srcKeyGroup = srcKeyPos->second;
        bool alreadyStored = false;
        SrcKeyPair srcKeyPair = getSrcKeyPair(calibStoreEventKey);
        alreadyWrittenIter = calibStoreTypesAlreadyWritten.find(srcKeyPair);
        if (alreadyWrittenIter != calibStoreTypesAlreadyWritten.end()) {
          const set<const type_info *> & typesWritten = alreadyWrittenIter->second;
          if (typesWritten.find(calibStoreEventKey.typeinfo()) != typesWritten.end()) {
            alreadyStored = true;
            MsgLog(logger,TRACELVL,"calibStore data for " << calibStoreEventKey
                   << " already stored. No need to store again for eventKey: " 
                   << calibEventKey);
          }
        }
        if (not alreadyStored) {
          try {
            srcKeyGroup.storeData(calibStoreEventKey, inCalibStore, evt, env);
          } catch (ErrSvc::Issue &issue) {
            m_calibStoreGroupDir.dump();
            MsgLog(logger,error, "caught exception trying to storeData for " << calibStoreEventKey << " issue: " << issue.what());
            throw issue;
          }
          alreadyWrittenIter = calibStoreTypesAlreadyWritten.find(srcKeyPair);
          if (alreadyWrittenIter == calibStoreTypesAlreadyWritten.end()) {
            set<const type_info *> typesWritten;
            typesWritten.insert(calibStoreEventKey.typeinfo());
            calibStoreTypesAlreadyWritten[srcKeyPair] = typesWritten;
          } else {
            set<const type_info *> &typesWritten = alreadyWrittenIter->second;
            typesWritten.insert(calibStoreEventKey.typeinfo());
          }
        }
      } else {
        MsgLog(logger, DBGLVL, " DID NOT get calib store data for type " 
               << PSEvt::TypeInfoUtils::typeInfoRealName(calibStoreType) 
               << " for " << calibEventKey);
      }
    }
  }
  // close things up
  m_calibStoreGroupDir.closeGroups();
  calibGroup.close();
  removeCalibStoreHdfWriters(m_hdfWriters);
}

herr_t H5Output::flushOutputFile() {
  if (m_h5file.valid()) {
    herr_t err = H5Fflush(m_h5file.id(), H5F_SCOPE_LOCAL);
    if (err < 0) MsgLog(logger, error, "error calling H5Fflush on outputfile: " << m_h5fileName
                        << " with id=" << m_h5file.id() << " err=" << err);
    return err;
  }
  MsgLog(logger, error, "flushOutputFile called but file is not valid");
  return -1;
}

  
void H5Output::endJob(Event& evt, Env& env) 
{
  boost::shared_ptr<EventId> eventId = evt.get();
  if (not m_splitScanMgr->isMPIMaster()) {
    lookForAndStoreEndData(evt, env, m_currentConfigureGroup, 
                           m_configureEndGroupDir, m_currentConfigureEndGroup);
  }
  if (not m_exclude_calibstore) {
    lookForAndStoreCalibData(evt, env, m_currentConfigureGroup);
  }
  m_configureGroupDir.closeGroups();
  m_epicsGroupDir.processEndJob();
  if (eventId) ::storeClock ( m_currentConfigureGroup, eventId->time(), "end" ) ;
  m_currentConfigureGroup.close();

  m_chunkManager.endJob();
  ++m_currentConfigureCounter;

  flushOutputFile();
  H5OpenObjects fileOpenObjects(m_h5file);
  if (fileOpenObjects.ALL - fileOpenObjects.FILE > 0) {
    bool detailedMsg = false;
    MsgLog(logger, trace, "endjob(). flushed output file. About to close :  " << m_h5fileName 
           << " but it has open objects: " << fileOpenObjects.dumpStr(detailedMsg));
    MsgLog(logger, trace, "Will close, and reopen before final close - this tends to clear open objects.");
    m_h5file.close();
    m_h5file = hdf5pp::File::open(m_h5fileName, hdf5pp::File::Read);
    H5OpenObjects fileReOpenObjects(m_h5file);
    if (fileReOpenObjects.ALL - fileReOpenObjects.FILE > 0) {
      bool detailedMsg = true;
      MsgLog(logger,info,"endJob() - issue with closing file " 
             << m_h5fileName << ", open hdf5 objects still exist after closing and reopening:"
             << fileReOpenObjects.dumpStr(detailedMsg));
      MsgLog(logger, info, "Will manually close open objects. " 
             << " may generate noisy error messages when Translator shuts down");
      fileReOpenObjects.closeOpenNonFileIds();
      flushOutputFile();
    }
  }
  // final close
  m_h5file.close();
    
  if (not m_quiet) {
    if (not m_splitScanMgr->isMPIMaster()) {
      // the driver will report timing when in mpi master mode
      reportRunTime();
    }
  }
}

void H5Output::reportRunTime() {
  m_endTime = LusiTime::Time::now();
  double deltaTime = (m_endTime.sec()-m_startTime.sec()) + (m_endTime.nsec()-m_startTime.nsec())/1e9;
  double rateHertz = double(m_totalEventsProcessed)/double(deltaTime);
  MsgLog(logger, info, "Translator + psana processing, real time (finish - start): " << deltaTime << " (sec) =  " 
         << deltaTime/60.0 << " (min)");
  MsgLog(logger, info, "Translator processing time: " << m_translatorTime << " (sec) = "
         << m_translatorTime/60.0 << " (min)");
  MsgLog(logger, info, "Translator proceesing as percentage of Translator+psana: " 
         << 100.0 * (m_translatorTime/deltaTime) << "%");
  MsgLog(logger, info, "Number of events processed: " << m_totalEventsProcessed);
  MsgLog(logger, info, "events/second (rate hertz): " << rateHertz);
  MsgLog(logger, info, "Number of Calib Cycles processed: " << m_totalCalibCyclesProcessed);
}

////////////////////////////////////////////////////
// shut down

H5Output::~H5Output() 
{
  MsgLog(logger, TRACELVL, "desctructor");
}

//////////////////////////////////////////////////////
// Helper functions to module event processing

void H5Output::createNextConfigureGroup(boost::shared_ptr<EventId> eventId) {
  char groupName[128];
  sprintf(groupName,"Configure:%4.4lu", m_currentConfigureCounter);
  m_currentConfigureGroup = m_h5file.createGroup(groupName);
  if (eventId) {
    ::storeClock ( m_currentConfigureGroup, eventId->time(), "start" );
  } else {
    MsgLog(logger,TRACELVL,name() << ": no eventId to store start.seconds, start.nanoseconds in configureGroup");
  }
  MsgLog(logger,TRACELVL,name() << ": createNextConfigureGroup: " << groupName);
}

void H5Output::createNextRunGroup(boost::shared_ptr<EventId> eventId) {
  char groupName[128];
  sprintf(groupName,"Run:%4.4lu", m_currentRunCounter);
  if (m_currentConfigureGroup.hasChild(groupName)) {
    ostringstream msg;
    msg << "createNextRunGroup: config group " 
        << m_currentConfigureGroup.name() 
        << " already has child: " << groupName;
    throw std::runtime_error(msg.str());
  }
  m_currentRunGroup = m_currentConfigureGroup.createGroup(groupName);
  MsgLog(logger,TRACELVL,"createNextRunGroup - created group. isValid=" << m_currentRunGroup.valid() << " group=" << m_currentRunGroup);
  if (eventId) {
    ::storeClock ( m_currentRunGroup, eventId->time(), "start" ) ;
  } else {
    MsgLog(logger, DBGLVL,name() << ": createNextRunGroup - no eventId to store start.seconds, start.nanoseconds");
  }
  MsgLog(logger, DBGLVL,name() << ": createNextRunGroup: " << groupName);
}

void H5Output::createNextCalibCycleGroup(boost::shared_ptr<EventId> eventId) {
  char groupName[128];
  sprintf(groupName,"CalibCycle:%4.4lu", m_currentCalibCycleCounter);
  if (not m_currentRunGroup.valid()) {
    MsgLog(logger, fatal, "createNextCalibCycleGroup: runGroup is invalid: group=" << m_currentRunGroup);
  } else { 
    MsgLog(logger, DBGLVL, name() << " run group is valid: " << m_currentRunGroup);
  }
  m_currentCalibCycleGroup = m_currentRunGroup.createGroup(groupName);
  if (eventId) {
    ::storeClock ( m_currentCalibCycleGroup, eventId->time(), "start" ) ;
  } else {
    MsgLog(logger,DBGLVL,name() << ": createNextCalibCycleGroup: no valid eventId to create start.seconds, nanoseconds");
  }
  MsgLog(logger,DBGLVL,name() << ": createNextCalibCycleGroup: " << groupName);
}


void H5Output::closeH5FileDueToEventException() {
  m_calibCycleGroupDir.closeGroups();
  m_runGroupDir.closeGroups();
  m_configureGroupDir.closeGroups();
  m_currentCalibCycleGroup.close();
  m_currentRunGroup.close();
  m_currentConfigureGroup.close();
  // flush everything for this file to disk - this is mostly for split scan mode - reading while writing
  flushOutputFile();
  m_h5file.close();
}

namespace {

  // key must already have any special prefixes such as translate_vlen striped from it.
  bool keyIsFiltered(const string & key, const bool includeAll, const bool isExclude, const set<string> & filterSet) {
    if (includeAll) return false;
    bool keyInFilterList = filterSet.find(key) != filterSet.end();
    if (isExclude) {
      bool inExcludeList = keyInFilterList;
      if (inExcludeList) {
        MsgLog(logger,TRACELVL,"keyIsFiltered: key= " << key << " is in exclude list - no translation");
        return true;
      } else {
        return false;
      }
    } 
    // it is an include set
    bool keyNotInIncludeSet = not keyInFilterList;
    if (keyNotInIncludeSet) {
      MsgLog(logger,TRACELVL,"keyIsFiltered: key=" << key << " is not listed in include list - no translation");
      return true;
    }
    return false;
  }

} // local namespace

bool H5Output::keyIsFiltered(const string &key) {
  string afterPrefix;
  bool retVal = ::keyIsFiltered(key, m_includeAllKey, m_keyFilterIsExclude, m_keyFilterSet);
  return retVal;
}

bool H5Output::fullEventKeyIsFiltered(const std::type_info *typeInfoPtr, const Pds::Src &src, const std::string &key) {
  if (m_includeAllEventKey) return false;
  bool eventKeyInList = false;
  std::map<const std::type_info *, SrcKeyList>::iterator typePos = m_eventKeyFilters.find(typeInfoPtr);
  if (typePos != m_eventKeyFilters.end()) {
    SrcKeyList &srcKeyList = typePos->second;
    for (SrcKeyList::iterator pos = srcKeyList.begin(); pos != srcKeyList.end(); ++pos) {
      if (pos->first.match(src) and (pos->second == key)) {
        eventKeyInList = true;
        break;
      }
    }
  }
  if ((not eventKeyInList) and (not m_eventkeyFilterIsExclude)) {
    return true;
  }
  if (eventKeyInList and m_eventkeyFilterIsExclude) {
    return true;
  }
  return false;
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
      MsgLog(logger,TRACELVL,"src: log=0x" << hex << src.log() << " phy=0x" << src.phy() 
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
    MsgLog(logger,TRACELVL,"src: log=0x" << hex << src.log() << " phy=0x" << src.phy() 
           << " name=" << m_h5groupNames->nameForSrc(src) 
           << " does not match any Source in include list - no translation");
    return true;
  }
  return false;
}

const DataSetCreationProperties & H5Output::lookUpDataSetCreationProp(TypeClass typeClass) {
  switch (typeClass) {
  case NdarrayType:
    return m_ndarrayCreateDsetProp;
  case StringType:
    return m_stringCreateDsetProp;
  default:
    return m_defaultCreateDsetProp;
  }
}

string H5Output::eventPosition() {
  stringstream res;
  res << "eventPosition: Configure:" << m_currentConfigureCounter
      << "/Run:" << m_currentRunCounter
      << "/CalibCycle:" << m_currentCalibCycleCounter
      << "/Event:" << m_currentEventCounter;
  return res.str();
}

const string H5Output::calibStoreGroupName("CalibStore");

PSANA_MODULE_FACTORY(H5Output);
