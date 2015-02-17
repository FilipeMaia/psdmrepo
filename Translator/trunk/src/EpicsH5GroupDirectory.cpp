#include <set>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>

#include "boost/make_shared.hpp"
#include "MsgLogger/MsgLogger.h"

#include "Translator/EpicsH5GroupDirectory.h"
#include "Translator/hdf5util.h"

using namespace Translator;
using namespace std;

namespace {

  const string logger(const string addTo="") {
    static const string epicsH5("Translator.EpicsH5");
    if (addTo.size()>0) {
      return epicsH5 + std::string(".") + addTo;
    }
    return epicsH5;
  }

  bool thereAreEpics(PSEnv::EpicsStore &epicsStore) {
    return epicsStore.pvNames().size()>0;
  }

  // normalize alias name, remove special characters
  std::string normAliasName(string alias) {
    string origAlias = alias;
    std::replace(alias.begin(), alias.end(), '/', '_');
    if (origAlias != alias) MsgLog(logger(),trace," normalized alias '" 
                                   << origAlias << "' to '" << alias << "'");
    return alias;
  }

  void createEpicsTypeAndSrcGroups(const hid_t parentGroup, const string & msgLogType, 
                                   hid_t & typeGroup, hid_t & srcGroup) {

    static const char * EPICS_TYPE_NAME = "Epics::EpicsPv";
    static const char * EPICS_SRC_NAME = "EpicsArch.0:NoDevice.0";
    static const uint64_t EPICS_SRC_VAL = 0x900000001004973;

    typeGroup = H5Gcreate(parentGroup, EPICS_TYPE_NAME, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (typeGroup < 0) MsgLog(logger(),fatal,"unable to create epics " 
                              << msgLogType << " type group. parentGroup=" << parentGroup);
    MsgLog(logger(),debug,"Created group " << EPICS_TYPE_NAME << " with id=" << typeGroup 
           << " as child of group " << parentGroup);
    srcGroup = H5Gcreate(typeGroup, EPICS_SRC_NAME, 
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (srcGroup < 0) MsgLog(logger(),fatal,"unable to create epics " << msgLogType << " src group");
    MsgLog(logger(),debug,"Created group " << EPICS_SRC_NAME << " with id " << srcGroup
           << " as child of group " << typeGroup);
    hdf5util::addAttribute_uint64(srcGroup, "_xtcSrc", EPICS_SRC_VAL);
    MsgLog(logger(),debug,"Added attribute _xtcSrc="<< EPICS_SRC_VAL << " to group " << srcGroup);
  }
  
  void createEpicsPvGroup(hid_t parentGroup, const string & pvName, 
                          const string & msgLogType, map<string, hid_t> &epicsPvGroups) {
    hid_t epicsPvGroup = H5Gcreate(parentGroup,pvName.c_str(),
                                   H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
    if (epicsPvGroup<0) {
      MsgLog(logger(),fatal,"failed to created " << msgLogType 
             << " epics pv group for " << pvName);
    }
    epicsPvGroups[pvName] = epicsPvGroup;
    MsgLog(logger(),debug,"created epics group for pvname=" << pvName << " id= " 
           << epicsPvGroup << " as child of group= " << parentGroup << " in " << msgLogType);
  }

  void createEpicsPvGroups(hid_t parentGroup, 
                           PSEnv::EpicsStore & epicsStore, const string msgLogType,
                           map<string, hid_t> &epicsPvGroups) {
    // create the epics pv name groups as children of the parent group
    vector<string> pvNames = epicsStore.pvNames();
    sort(pvNames.begin(),pvNames.end());
    for (unsigned i = 0; i < pvNames.size(); ++i) {
      string &pvName = pvNames[i];
      if (i > 0 and pvNames[i-1] == pvName) {
        // this should never happen - epicsStore keys based on the pvname
        MsgLog(logger(), warning, "**pvName " << pvName << " shows up more than once in epics store");
        continue;
      }
      createEpicsPvGroup(parentGroup,pvName,msgLogType,epicsPvGroups);
    }
  }

  /* ----------------
     createBeginJobAliasLinksAndGetAliasesForEvents - 
     This is called during create job. It creates the alias links at that point, for the 
     ctrl headers. It also fills the argument pv2aliases with all the aliases for a given 
     pv. There can be more than one alias for a given pv. When a pv first shows up during the 
     event data, pv2aliases should be queried to see if any aliases should be created for the
     data.
     --------------- */
  void createBeginJobAliasLinksAndGetAliasesForEvents(hid_t parentGroup, PSEnv::EpicsStore & epicsStore, 
                        map<string, vector<string> > & pv2aliases, const string msgLogType) {

    vector<string> validTargetsVector = epicsStore.pvNames();
    set<string> validTargets;
    for (unsigned int i = 0; i < validTargetsVector.size(); ++i) validTargets.insert(validTargetsVector.at(i));

    typedef pair<string,string> NormOrigAliasPair;
    vector<NormOrigAliasPair> normOrigAliasPairs;
    vector<string> aliases = epicsStore.aliases();
    for (size_t idx = 0; idx < aliases.size(); ++idx) {
      string origAlias = aliases[idx];
      string normAlias = normAliasName(aliases[idx]);
      size_t n = normAlias.size();
      if (n>0 and (normAlias[0] == ' ' or normAlias[n-1]==' ')) {
        MsgLog(logger(),warning,"Normalized epics alias has a leading or trailing blank: '" << normAlias << "'");
      }
      normOrigAliasPairs.push_back( pair<string,string>(normAlias, origAlias) );
    }
    pv2aliases.clear();
    sort(normOrigAliasPairs.begin(), normOrigAliasPairs.end());
    for (size_t idx = 0; idx < normOrigAliasPairs.size(); ++idx) {
      string normAlias = normOrigAliasPairs[idx].first;
      if ( (idx>0) and (normAlias == normOrigAliasPairs[idx-1].first) ) {
        MsgLog(logger(), warning, 
               " duplicate normalized alias name: " << normAlias << " skipping");
        continue;
      }
      string targetName = epicsStore.pvName(normOrigAliasPairs[idx].second);
      if (targetName.size()==0) {
        MsgLog(logger(), warning, "norm alias '" << normAlias << "' has an empty target");
        continue;
      }
      if (validTargets.find(targetName)==validTargets.end()) {
        MsgLog(logger(), warning, "alias '" << normOrigAliasPairs[idx].second << 
               "' has a target: '" << targetName << "' that is not a valid pvname");
        continue;
      }
      if (normAliasName(targetName) == normAlias) {
        MsgLog(logger(), debug, "normAlias '" << normAlias << "' is the same as normalized targetname, skipping.");
        continue;
      } 
      if (validTargets.find(normAlias) != validTargets.end()) {
        MsgLog(logger(), warning, "normalized alias '" << normAlias 
               << "' (which points to '"<< targetName << "') is the same as existing group. skipping.");
        continue;
      }

      herr_t err = H5Lcreate_soft(targetName.c_str(), parentGroup, normAlias.c_str(), 
                                  H5P_DEFAULT, H5P_DEFAULT);
      if ( err < 0 ) {
        MsgLog(logger(), warning, "H5Lcreate_soft failed for norm alias= '" 
               << normAlias << "' target= '" << targetName 
               << "' relative to epics src " << msgLogType << " group hid=" << parentGroup);
      } else {
        pv2aliases[targetName].push_back(normAlias);
        MsgLog(logger(),debug,"Created alias link: norm alias '" << normAlias << "' target '"
               << targetName << "' parentGroup= " << parentGroup);
      }
    }
  }

  void pvIdReport(ostream & str, vector<int16_t> & pvIds, const int totalPvNames) {
    // we expect to create one group per pvname, and we generally expect the associated
    // pvIds to go one up from 0 to the number of pv names.  However sometimes the
    // same pvname shows up 2 or more times with different pv id's.  The pv data should
    // be identicial, but we do not check this.  This function reports on any gaps
    // in the pvIds.  We expect these missing pvId's to come from duplicate's to the
    // pvnames that we wrote.
    map<int16_t,int> pvIdCounts;
    int16_t minPvId = pvIds.at(0);
    int16_t maxPvId = pvIds.at(0);
    for (unsigned int i = 0; i < pvIds.size(); ++i) {
      int16_t pvId = pvIds.at(i);
      minPvId = std::min(minPvId,pvId);
      maxPvId = std::max(maxPvId,pvId);
      if (pvIdCounts.find(pvId) != pvIdCounts.end()) {
        ++pvIdCounts[pvId];
      } else {
        pvIdCounts[pvId] = 1;
      }
    }
    set<uint16_t> missingPvIds, dupPvIds;
    for (uint16_t i = 0; i < totalPvNames; ++i) missingPvIds.insert(i);
    for (map<int16_t,int>::iterator pos = pvIdCounts.begin(); pos != pvIdCounts.end(); ++pos) {
      uint16_t pvId = pos->first;
      int count = pos->second;
      missingPvIds.erase(pvId);
      if (count>1) dupPvIds.insert(pvId);
    }
    str << "created " << pvIds.size() << " epics groups.";
    str << " The PvIds are in [" << minPvId <<", "<< maxPvId<<"].";
    str <<" There were " << totalPvNames << " pv names.";
    if (missingPvIds.size() > 0) {
      str << " Normally pvIds range from [0," << totalPvNames-1 << "].";
      str << " These pvId's were not seen:";
      for (set<uint16_t>::iterator missingPos = missingPvIds.begin();
           missingPos != missingPvIds.end(); ++missingPos) {
        str << " " << *missingPos;
      }
      str << ". The missing pvId's are expected to be present in the xtc files, but overwritten in the Psana ";
      str << " epics store. They are expected to have the same data as what is in the epics store (but we ";
      str << "do not verify this).";
    }
    if (dupPvIds.size()>0) {
      str << " These pv id's showed up more than once: ";
      for (set<uint16_t>::iterator dupPos = dupPvIds.begin();
           dupPos != dupPvIds.end(); ++dupPos) {
        str << " " << *dupPos;
      }
      str << " happens with multiple Source's for epics.**";
    }
  }

  void reportEpics(ostream & str, PSEnv::EpicsStore & epicsStore) {
    vector<string> pvNames = epicsStore.pvNames();
    str << "** EPICS REPORT **" << endl;
    str << pvNames.size() << " pvnames in epics store - names and pvids: " << endl;
    for (unsigned i = 0; i < pvNames.size(); ++i) {
      boost::shared_ptr<Psana::Epics::EpicsPvCtrlHeader> pvCtrl = epicsStore.getPV(pvNames.at(i));
      str << "pvId=" << pvCtrl->pvId() << " pvname=" << pvNames.at(i) << endl;
    }
  }

} // local namespace

EpicsH5GroupDirectory::EpicsH5GroupDirectory() :
  m_epicsStoreMode(Unknown),
  m_configureGroup(-1),
  m_currentCalibCycleGroup(-1),
  m_configEpicsTypeGroup(-1),
  m_configEpicsSrcGroup(-1),
  m_calibCycleEpicsTypeGroup(-1),
  m_calibCycleEpicsSrcGroup(-1),
  m_epicsStatus(unknown)
{}

void EpicsH5GroupDirectory::initialize(EpicsStoreMode epicsStoreMode,
                                       boost::shared_ptr<HdfWriterEventId> hdfWriterEventId,
                                       const DataSetCreationProperties & oneElemEpicsPvCreateDsetProp,
                                       const DataSetCreationProperties & manyElemEpicsPvCreateDsetProp)
{
  m_epicsStoreMode = epicsStoreMode;
  if (m_epicsStoreMode == Unknown) MsgLog(logger(), fatal, "epics store mode is unknown");
  m_hdfWriterEpicsPv = boost::make_shared<HdfWriterEpicsPv>(oneElemEpicsPvCreateDsetProp, 
                                                            manyElemEpicsPvCreateDsetProp, 
                                                            hdfWriterEventId);
}

bool EpicsH5GroupDirectory::checkIfStoringEpics() {
  if (m_epicsStoreMode == Unknown) MsgLog(logger(), fatal, "epics store mode is unknown");
  if (m_epicsStoreMode == DoNotStoreEpics) return false;
  return true;
}

void EpicsH5GroupDirectory::processBeginJob(hid_t currentConfigGroup, 
                                            PSEnv::EpicsStore &epicsStore,
                                            boost::shared_ptr<PSEvt::EventId> eventId) 
{
  if (not checkIfStoringEpics()) return;
  if (not thereAreEpics(epicsStore)) {
    m_epicsStatus = noEpics;
    return;
  }
  m_epicsStatus = hasEpics;
  m_configureGroup = currentConfigGroup;
  createEpicsTypeAndSrcGroups(m_configureGroup, "config",  
                              m_configEpicsTypeGroup, m_configEpicsSrcGroup);
  WithMsgLog(logger(),debug,str) {
    reportEpics(str,epicsStore);
  }
  createEpicsPvGroups(m_configEpicsSrcGroup, epicsStore, "config", 
                      m_configEpicsPvGroups);
  createBeginJobAliasLinksAndGetAliasesForEvents(m_configEpicsSrcGroup, epicsStore, 
                                                 m_epicsPv2Aliases, "config");

  // create the datasets for the configure epics groups.  
  vector<string> pvNames = epicsStore.pvNames();
  vector<int16_t> pvIds;
  for (size_t pvIdx = 0; pvIdx < pvNames.size(); ++pvIdx) {
    const string &pvName = pvNames[pvIdx];
    // we only do ctrl headers here
    boost::shared_ptr<Psana::Epics::EpicsPvCtrlHeader> pvCtrl = epicsStore.getPV(pvName);
    if (not pvCtrl) {
      MsgLog(logger(), warning, "pvName " << pvName 
             << " does not have control header during configure");
      continue;
    }
    map<string,hid_t>::iterator groupIdPos = m_configEpicsPvGroups.find(pvName);
    if (groupIdPos == m_configEpicsPvGroups.end()) {
      MsgLog(logger(),fatal,"event loop, unexpected pvName: " << pvName << " not in config group map");
    }
    hid_t groupId = groupIdPos->second;
    int16_t dbrType = pvCtrl->dbrType();
    pvIds.push_back(pvCtrl->pvId());
    try {
      m_hdfWriterEpicsPv->oneTimeCreateAndWrite(groupId, dbrType, 
                                                epicsStore, pvName, eventId);
    } catch (HdfWriterEpicsPv::Exception & except) {
      MsgLog(logger(),fatal, "beginJob" << except.what());
    }
  }
  // produce a trace message to audit any missing pvid's. The pvname should
  // uniquely identify the pv, however xtc files can have several pv's with 
  // the same name, but different id's.  Presumably they all hold the same
  // data and only one need be recorded.  The trace message allows the user
  // to investigate this further.
  WithMsgLog(logger("processBeginJob"),trace,str) {
    pvIdReport(str,pvIds, pvNames.size());
  }
}

void EpicsH5GroupDirectory::processBeginCalibCycle(hid_t currentCalibCycleGroup, 
                                                   PSEnv::EpicsStore &epicsStore) {
  if (not checkIfStoringEpics()) return;
  if (m_epicsStatus == unknown) {
    MsgLog(logger("processBeginCalibCycle"), fatal, 
           "epicsStatus not set, processBeginJob not called");
  }
  if (thereAreEpics(epicsStore) and (m_epicsStatus == noEpics)) {
    MsgLog(logger("processBeginCalibCycle"), warning, 
           "No epics detected during beginJob, but epics detected "
           << "during calib cycle. aliases and epics ctrl pv's "
           << " will not be written");
    m_epicsStatus = hasEpics;
  }
  if (m_epicsStatus == noEpics) return;

  m_currentCalibCycleGroup = currentCalibCycleGroup;
  m_epicsTypeAndSrcGroupsCreatedForThisCalibCycle = false;
}

void EpicsH5GroupDirectory::processEvent(PSEnv::EpicsStore & epicsStore, 
                                         boost::shared_ptr<PSEvt::EventId> eventId) {
  if (not checkIfStoringEpics()) return;
  vector<string> pvNames = epicsStore.pvNames();
  vector<string>::iterator pvIter;
  const PSEnv::EpicsStoreImpl& epicsStoreImpl = epicsStore.internal_implementation();
  for (pvIter = pvNames.begin(); pvIter != pvNames.end(); ++pvIter) {
    string &pvName = *pvIter;
    // we only expect time epics pv's
    PSEnv::EpicsStoreImpl::TimeHeaderAndEventTag timeHeaderAndEventTag = 
      epicsStoreImpl.getTimeAndEventTag(pvName);
    boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader> pvTime = timeHeaderAndEventTag.pv;
    long pvTimeEventTag = timeHeaderAndEventTag.eventTag;
    if (not pvTime) {
      boost::shared_ptr<Psana::Epics::EpicsPvHeader> pvHdr = epicsStore.getPV(pvName);
      if (not pvHdr) {
        MsgLog(logger(), error, "no EpicsPvHeader associated with epics pv: " << pvName);
      } else {
        if (not pvHdr->isCtrl()) {
          MsgLog(logger(), warning, "epics writing only implement for Time pv's, but pv: " 
                 << pvName << " is neither Time (nor Ctrl) pv will not be written. dbr is "
                 <<  pvHdr->dbrType());
        }
      }
      continue;
    }
    ostringstream debugMsg;
    debugMsg << pvName;
    //    const Psana::Epics::epicsTimeStamp &pvStamp = pvTime->stamp();
    bool createDataset = false;
    bool appendToDataset = false;
    map<string, long>::iterator lastPvEventTagPos = m_lastPvEventTag.find(pvName);
    if (lastPvEventTagPos == m_lastPvEventTag.end()) {
      createDataset = true;
      appendToDataset = true;
      m_lastPvEventTag[pvName] = -1;
      lastPvEventTagPos = m_lastPvEventTag.find(pvName);
      debugMsg << ", first event data - createDataset=true, appendToDataset=true";
    } else {
      if (writeCurrentPv(pvTimeEventTag, lastPvEventTagPos->second)) {
        debugMsg << ", pv event tag has changed (or we are writing on every event), appendTODataset=true";
        appendToDataset = true;
      } else {
        debugMsg << ", previously seen and no change to tag";
      }
    }
    if (not appendToDataset) {
      MsgLog(logger(), debug, debugMsg.str() << ", skipping, not appending data");
      continue;
    }
    MsgLog(logger(), debug, debugMsg.str() << ", going to translate");
    if (not m_epicsTypeAndSrcGroupsCreatedForThisCalibCycle) {
      createEpicsTypeAndSrcGroups(m_currentCalibCycleGroup, "calib",  
                                  m_calibCycleEpicsTypeGroup, m_calibCycleEpicsSrcGroup);
      m_epicsTypeAndSrcGroupsCreatedForThisCalibCycle = true;
    }
    map<string,hid_t>::iterator groupIdPos = m_calibEpicsPvGroups.find(pvName);
    if (groupIdPos == m_calibEpicsPvGroups.end()) {
      createEpicsPvGroup(m_calibCycleEpicsSrcGroup, 
                         pvName, "calib",  m_calibEpicsPvGroups);
      createDataset = true;
      std::map<std::string, std::vector<std::string> >::iterator pv2AliasPos;
      pv2AliasPos = m_epicsPv2Aliases.find(pvName);
      if (pv2AliasPos != m_epicsPv2Aliases.end()) {
        vector<string> & aliasesForThisPv = pv2AliasPos->second;
        for (unsigned idx = 0; idx < aliasesForThisPv.size(); ++idx) {
          const string & alias = aliasesForThisPv.at(idx);
          herr_t err = H5Lcreate_soft(pvName.c_str(), 
                                      m_calibCycleEpicsSrcGroup,
                                      alias.c_str(),
                                      H5P_DEFAULT, H5P_DEFAULT);
          if (err < 0) MsgLog(logger(), warning, "H5Lcreate_soft call failed for "
                              << "alias= '" << alias << "' target= '"
                              << pvName << " during a calib cycle");
        }
      }
      groupIdPos = m_calibEpicsPvGroups.find(pvName);
    }
    int16_t dbrType = pvTime->dbrType();
    hid_t groupId = groupIdPos->second;
    try {
      if (createDataset) {
        m_hdfWriterEpicsPv->createAndAppend(groupId, dbrType, epicsStore, pvName, eventId);
      } else {
        m_hdfWriterEpicsPv->append(groupId, dbrType, epicsStore, pvName, eventId);
      }
    } catch (HdfWriterEpicsPv::Exception &except) {
      cout << except.what() << endl;
      MsgLog(logger(),fatal,"error with create and or append: " << debugMsg.str());
    }
    lastPvEventTagPos->second = pvTimeEventTag;
  }
}

bool EpicsH5GroupDirectory::writeCurrentPv(long currentEventTag, long previousEventTag) {
  if (m_epicsStoreMode == StoreAllEpicsOnEveryShot) return true;
  if ((previousEventTag < 0) or (currentEventTag < 0)) return true;
  if (currentEventTag != previousEventTag) return true;
  return false;
}

void EpicsH5GroupDirectory::processEndCalibCycle() {
  if (not checkIfStoringEpics()) return;
  std::map<std::string, hid_t>::iterator iter;
  for (iter = m_calibEpicsPvGroups.begin(); iter != m_calibEpicsPvGroups.end(); ++iter) {
    hid_t &epicsPvGroup = iter->second;
    try {
      m_hdfWriterEpicsPv->closeDataset(epicsPvGroup);
    } catch (HdfWriterGeneric::GroupMapException &issue) {
      MsgLog(logger(),trace, issue.what() <<
             " from processEndCalibCycle, most likely no data written for epics pv: " << 
             iter->first);
    }
    herr_t res = H5Gclose(epicsPvGroup);
    if (res<0) MsgLog(logger(),fatal,"failed to close epics pv calib group " << iter->first);
  }
  m_calibEpicsPvGroups.clear();
  if (m_calibCycleEpicsSrcGroup>=0) {
    herr_t res = H5Gclose(m_calibCycleEpicsSrcGroup);
    if (res<0) MsgLog(logger(),fatal,"failed to close calib epics src group");
    m_calibCycleEpicsSrcGroup = -1;
  }
  if (m_calibCycleEpicsTypeGroup>=0) {
    herr_t res = H5Gclose(m_calibCycleEpicsTypeGroup);
    if (res<0) MsgLog(logger(),fatal,"failed to close calib epics type group");
    m_calibCycleEpicsTypeGroup = -1;
  }
  m_currentCalibCycleGroup = -1;
}

void EpicsH5GroupDirectory::processEndJob() {
  if (not checkIfStoringEpics()) return;
  if ((m_epicsStoreMode == RepeatEpicsEachCalib) or 
      (m_epicsStoreMode == StoreAllEpicsOnEveryShot)) 
  {
    m_lastPvEventTag.clear();
  }
  m_configEpicsPvGroups.clear();
  if (m_configEpicsSrcGroup>=0) {
    herr_t res = H5Gclose(m_configEpicsSrcGroup);
    if (res<0) MsgLog(logger(),fatal,"failed to close config epics src group");
    m_configEpicsSrcGroup = -1;
  }
  if (m_configEpicsTypeGroup>=0) {
    herr_t res = H5Gclose(m_configEpicsTypeGroup);
    if (res<0) MsgLog(logger(),fatal,"failed to close config epics type group");
    m_configureGroup = -1;
  }
  m_configureGroup = -1;
}

std::string EpicsH5GroupDirectory::epicsStoreMode2str(const EpicsStoreMode storeMode) {
  if (storeMode == DoNotStoreEpics) return string("DoNotStoreEpics");
  if (storeMode == RepeatEpicsEachCalib) return string("RepeatEpicsEachCalib");
  if (storeMode == OnlyStoreEpicsUpdates) return string("OnlyStoreEpicsUpdates");
  if (storeMode == StoreAllEpicsOnEveryShot) return string("StoreAllEpicsOnEveryShot");
  if (storeMode == Unknown) return string("Unknown");
  return string("**Invalid**");
}
