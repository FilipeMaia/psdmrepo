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

  void createEpicsTypeAndSrcGroups(const hid_t parentGroup, const string msgLogType, 
                                   hid_t & typeGroup, hid_t & srcGroup) {

    static const char * EPICS_TYPE_NAME = "Epics::EpicsPv";
    static const char * EPICS_SRC_NAME = "EpicsArch.0:NoDevice.0";
    static const uint64_t EPICS_SRC_VAL = 0x900000001004973;

    typeGroup = H5Gcreate(parentGroup, EPICS_TYPE_NAME, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (typeGroup < 0) MsgLog(logger(),fatal,"unable to create epics " << msgLogType << " type group");
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
  }

  void createAliasLinks(hid_t parentGroup, PSEnv::EpicsStore & epicsStore, 
                        const string msgLogType) {

    vector<string> validTargetsVector = epicsStore.pvNames();
    set<string> validTargets;
    for (unsigned int i = 0; i < validTargetsVector.size(); ++i) validTargets.insert(validTargetsVector.at(i));

    vector<string> aliases = epicsStore.aliases();
    for (size_t idx = 0; idx < aliases.size(); ++idx) {
      aliases[idx]=normAliasName(aliases[idx]);
      size_t n = aliases[idx].size();
      if (n>0 and (aliases[idx][0] == ' ' or aliases[idx][n-1]==' ')) {
        MsgLog(logger(),warning,"Epics alias has a leading or trailing blank: '" << aliases[idx] << "'");
      }
    }
    sort(aliases.begin(), aliases.end());
    for (size_t idx = 0; idx < aliases.size(); ++idx) {
      string alias = aliases[idx];
      if (idx>0 and alias==aliases[idx-1]) {
        MsgLog(logger(), warning, 
               " normalized alias name is the same as another PV or alias name: '" << alias
               << "' skipping this alias");
        continue;
      }
      string targetName = epicsStore.pvName(alias);
      if (targetName.size()==0) {
        MsgLog(logger(), warning, "alias '" << alias << "' has an empty target");
        continue;
      }
      if (validTargets.find(targetName)==validTargets.end()) {
        MsgLog(logger(), warning, "alias '" << alias << 
               "' has a target: '"<<targetName<<"' that is not a valid pvname");
        continue;
      }
      if (normAliasName(targetName) == alias) {
        MsgLog(logger(), debug, "alias '" << alias << "' is the same as normalized targetname, skipping.");
        continue;
      } 
      if (validTargets.find(alias) != validTargets.end()) {
        MsgLog(logger(), warning, "normalized alias '" << alias << "' (which points to '"<< targetName << "') is the same as existing group. skipping.");
        continue;
      }

      herr_t err = H5Lcreate_soft(targetName.c_str(), parentGroup, alias.c_str(), 
                                  H5P_DEFAULT, H5P_DEFAULT);
      if ( err < 0 ) {
        MsgLog(logger(), warning, "H5Lcreate_soft failed for alias= '" 
               << alias << "' target= '" << targetName 
               << "' relative to epics src " << msgLogType << " group hid=" << parentGroup);
      } else {
        MsgLog(logger(),debug,"Created alias link: alias '" << alias << "' target '"<<targetName<<
               "' parentGroup= " << parentGroup);
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
      str << " **this should not happen.**";
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
  m_configureGroup(-1),
  m_currentCalibCycleGroup(-1),
  m_configEpicsTypeGroup(-1),
  m_configEpicsSrcGroup(-1),
  m_calibCycleEpicsTypeGroup(-1),
  m_calibCycleEpicsSrcGroup(-1),
  m_epicsStatus(unknown)
{}

void EpicsH5GroupDirectory::initialize(boost::shared_ptr<HdfWriterEventId> hdfWriterEventId,
                                       const DataSetCreationProperties & epicsPvCreateDsetProp) 
{
  m_hdfWriterEpicsPv = boost::make_shared<HdfWriterEpicsPv>(epicsPvCreateDsetProp, hdfWriterEventId);
}

void EpicsH5GroupDirectory::processBeginJob(hid_t currentConfigGroup, 
                                            PSEnv::EpicsStore &epicsStore,
                                            boost::shared_ptr<PSEvt::EventId> eventId) 
{
  if (currentConfigGroup<0) MsgLog(logger(), fatal, "processBeginJob passed invalid group");
  m_configureGroup = currentConfigGroup;
  if (not thereAreEpics(epicsStore)) {
    m_epicsStatus = noEpics;
    return;
  }
  m_epicsStatus = hasEpics;

  createEpicsTypeAndSrcGroups(m_configureGroup, "config",  
                              m_configEpicsTypeGroup, m_configEpicsSrcGroup);
  WithMsgLog(logger(),debug,str) {
    reportEpics(str,epicsStore);
  }
  createEpicsPvGroups(m_configEpicsSrcGroup, epicsStore, "config", 
                      m_configEpicsPvGroups);
  createAliasLinks(m_configEpicsSrcGroup, epicsStore, "config");

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
  if (m_epicsStatus == unknown) {
    MsgLog(logger("processBeginCalibCycle"), fatal, 
           "epicsStatus not set, processBeginJob not called");
  }
  if (thereAreEpics(epicsStore) and (m_epicsStatus == noEpics)) {
    MsgLog(logger("processBeginCalibCycle"), warning, 
           "*unexpected DAQ problem*: no epics detected during beginJob, "
           << "but epics detected during calib cycle. epics is *NOT* being converted");
  }
  if (m_epicsStatus == noEpics) return;

  m_currentCalibCycleGroup = currentCalibCycleGroup;
  createEpicsTypeAndSrcGroups(m_currentCalibCycleGroup, "calib",  
                              m_calibCycleEpicsTypeGroup, m_calibCycleEpicsSrcGroup);
  createEpicsPvGroups(m_calibCycleEpicsSrcGroup, epicsStore, "calib", 
                      m_calibEpicsPvGroups);
  createAliasLinks(m_calibCycleEpicsSrcGroup, epicsStore, "calib");
}

void EpicsH5GroupDirectory::processEvent(PSEnv::EpicsStore & epicsStore, 
                                         boost::shared_ptr<PSEvt::EventId> eventId) {
  vector<string> pvNames = epicsStore.pvNames();
  vector<string>::iterator pvIter;
  for (pvIter = pvNames.begin(); pvIter != pvNames.end(); ++pvIter) {
    string &pvName = *pvIter;
    // we only expect time epics pv's
    boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader> pvTime = epicsStore.getPV(pvName);
    if (not pvTime) {
      MsgLog(logger(), debug, "processEvent - no time header for pv " << pvName);
      continue;
    }
    ostringstream debugMsg;
    debugMsg << pvName;
    const Psana::Epics::epicsTimeStamp &pvStamp = pvTime->stamp();
    bool createDataset = false;
    bool appendToDataset = false;
    map<string, Unroll::epicsTimeStamp>::iterator lastWriteTime;
    lastWriteTime = m_lastWriteMap.find(pvName);
    if (lastWriteTime == m_lastWriteMap.end()) {
      createDataset = true;
      appendToDataset = true;
      m_lastWriteMap[pvName] = Unroll::epicsTimeStamp();
      lastWriteTime = m_lastWriteMap.find(pvName);
      debugMsg << ", first event data - createDataset=true, appendToDataset=true";
    } else {
      if ((pvStamp.sec() != lastWriteTime->second.secPastEpoch) or (pvStamp.nsec() != lastWriteTime->second.nsec)) {
        debugMsg << ", previously seen, new timestamp, appendTODataset=true";
        appendToDataset = true;
      } else {
        debugMsg << ", previously seen and same timestamp";
      }
    }
    if (not appendToDataset) {
      MsgLog(logger(), debug, debugMsg.str() << ", skipping, not appending data");
      continue;
    }
    MsgLog(logger(), debug, debugMsg.str() << ", going to translate");
    int16_t dbrType = pvTime->dbrType();
    map<string,hid_t>::iterator groupIdPos = m_calibEpicsPvGroups.find(pvName);
    if (groupIdPos == m_calibEpicsPvGroups.end()) {
      MsgLog(logger(),fatal,"event loop, unexpected pvName: " << pvName << " not in group map");
    }
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
    lastWriteTime->second.secPastEpoch = pvStamp.sec();
    lastWriteTime->second.nsec = pvStamp.nsec();
  }
}

void EpicsH5GroupDirectory::processEndCalibCycle() {
  m_lastWriteMap.clear();  // this means we may copy the last epics pv from the 
                           // previous calib cycle to the start of the new calib cycle.
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

