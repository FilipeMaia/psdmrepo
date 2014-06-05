//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsStoreImpl...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSEnv/EpicsStoreImpl.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "EpicsStore";
  
  // time diff in seconds between EPICS epoch and UNIX epoch
  unsigned sec_1970_to_1990 = (20*365 + 5)*24*3600;

  std::string pvId2str(const std::tr1::tuple<uint32_t, uint32_t, int> &pvId) {
    std::stringstream o;
    o << "src.log=0x" << std::hex << std::tr1::get<0>(pvId)
      << " src.phy=0x" << std::hex << std::tr1::get<1>(pvId)
      << " pvId=" << std::dec << std::tr1::get<2>(pvId);
    return o.str();
  }
    

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSEnv {

//----------------
// Constructors --
//----------------
EpicsStoreImpl::EpicsStoreImpl ()
  : m_id2name()
  , m_name2id()
  , m_id2alias()
  , m_alias2id()
  , m_ctrlMap()
  , m_timeMap()
{
}

//--------------
// Destructor --
//--------------
EpicsStoreImpl::~EpicsStoreImpl ()
{
}

/// Store EPICS PV
void 
EpicsStoreImpl::store(const boost::shared_ptr<Psana::Epics::EpicsPvHeader>& pv, const Pds::Src& src, 
                      const std::string *pvName, long eventTag)
{
  PvId pvid(src.log(), src.phy(), pv->pvId());

  if (pv->isTime()) {

    // set name from passed in pvName, or find a name, or build fictional one
    std::string name;
    if (pvName != NULL) { 
      name = *pvName;
      MsgLog(logger, debug, "EpicsStore::store - storing TIME PV with passed in name=" << *pvName);
    } else {
      ID2Name::const_iterator it = m_id2name.find(pvid);
      if (it != m_id2name.end()) {
        name = it->second;
      } else {
        name = "PV:pvId=" + boost::lexical_cast<std::string>(pv->pvId()) +
          ":src_log=" + boost::lexical_cast<std::string>(src.log()) +
          ":src_phy=" + boost::lexical_cast<std::string>(src.phy());
        m_id2name.insert(std::make_pair(pvid, name));
        m_name2id.insert(std::make_pair(name, pvid));
      }
      MsgLog(logger, debug, "EpicsStore::store - storing TIME PV with id=" << pv->pvId());
    }
    boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader> tpv =
        boost::static_pointer_cast<Psana::Epics::EpicsPvTimeHeader>(pv);
    bool eventTagSpecified = eventTag >= 0;
    if (eventTagSpecified) {
      TimeMap::iterator pos = m_timeMap.find(name);
      if (pos != m_timeMap.end()) {
        TimeTagValue &previousTimeTag = pos->second;
        long previousEventTag = previousTimeTag.second;
        if (previousEventTag == eventTag) {
          // check stamps
          boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader> previousTimePV = previousTimeTag.first;
          const Psana::Epics::epicsTimeStamp previousStamp = previousTimePV->stamp();
          const Psana::Epics::epicsTimeStamp currentStamp = tpv->stamp();
          bool currentIsMoreRecent = ((currentStamp.sec() > previousStamp.sec()) or
                                      ((currentStamp.sec() == previousStamp.sec()) and 
                                       (currentStamp.nsec() > previousStamp.nsec())));
          if (currentIsMoreRecent) {
            m_timeMap[name] = TimeTagValue(tpv,eventTag);
          }
        }
      }
    } else {
      // not eventTagSpecified
      m_timeMap[name] = TimeTagValue(tpv,eventTag);
    }

  } else if (pv->isCtrl()) {

    MsgLog(logger, debug, "EpicsStore::store - storing CTRL PV with id=" << pv->pvId());
    boost::shared_ptr<Psana::Epics::EpicsPvCtrlHeader> ctrl =
        boost::static_pointer_cast<Psana::Epics::EpicsPvCtrlHeader>(pv);
    std::string name = ctrl->pvName();
    Name2ID::iterator matchingAliasPos = m_alias2id.find(name);
    bool aliasNameFoundThatIsThisPvName = (matchingAliasPos != m_alias2id.end());
    if ( aliasNameFoundThatIsThisPvName ) {
      MsgLog(logger, trace, "EpicsStore::store - alias: " << name
             << " is also used for a pvName. Discarding its use as an alias");
      PvId pvId = matchingAliasPos->second;
      ID2Name::iterator matchingAliasIdPos = m_id2alias.find(pvId);
      m_alias2id.erase(matchingAliasPos);
      if (matchingAliasIdPos != m_id2alias.end()) {
        m_id2alias.erase(matchingAliasIdPos);
      } else {
        MsgLog(logger,debug, "EpicsStore::store - while removing alias " << name
               << " could not find it's pvId: " << ::pvId2str(pvId)
               << " in the id2alias map. Unexpected");
      }
    }
      
    m_id2name.insert(std::make_pair(pvid, name));
    m_name2id.insert(std::make_pair(name, pvid));
    m_ctrlMap[name] = ctrl;
    
  } else {
    
    MsgLog(logger, warning, "EpicsStore::store - unexpected PV type: ID=" << pv->pvId() << " type=" << pv->dbrType());
    
  }
}


/// Store alias name for EPICS PV.
void
EpicsStoreImpl::storeAlias(const Pds::Src& src, int pvId, const std::string& alias)
{
  bool aliasIsAlsoPvName = m_name2id.find(alias) != m_name2id.end();
  if (aliasIsAlsoPvName) {
    MsgLog(logger, trace, "EpicsStore::storeAlias - alias: " << alias 
           << " is also used for a pvName. Discarding its use as an alias");
    return;
  }
  PvId pvid(src.log(), src.phy(), pvId);
  ID2Name::iterator pvIdPos = m_id2alias.find(pvid);
  bool pvIdAlreadStored = (pvIdPos != m_id2alias.end());
  if (pvIdAlreadStored) {
    const std::string &previousAlias = pvIdPos->second;
    if (previousAlias != alias) {
      MsgLog(logger,trace,"EpicsStore::storeAlias - pvId " << ::pvId2str(pvid)
             << " has already been used with alias: " << previousAlias
             << ". Attempt to store new alias: " << alias
             << ". The new alias will not be stored.");
    }
    return;
  }
  m_alias2id[alias] = pvid;
  m_id2alias[pvid] = alias;
}


/// Get the full list of PV names and aliases
void
EpicsStoreImpl::names(std::vector<std::string>& names) const
{
  names.clear();
  names.reserve(m_name2id.size() + m_alias2id.size());
  for (Name2ID::const_iterator it = m_name2id.begin(); it != m_name2id.end(); ++ it) {
    names.push_back(it->first);
  }
  for (Name2ID::const_iterator it = m_alias2id.begin(); it != m_alias2id.end(); ++ it) {
    names.push_back(it->first);
  }
  // check for pv names that may have come directly through the store routine:
  for (TimeMap::const_iterator it = m_timeMap.begin(); it != m_timeMap.end(); ++ it) {
    bool nameAlreadyAdded = (m_name2id.find(it->first) != m_name2id.end());
    if (not nameAlreadyAdded) {
      names.push_back(it->first);
    }
  }
}

/// Get the list of PV names
void
EpicsStoreImpl::pvNames(std::vector<std::string>& names) const
{
  names.clear();
  names.reserve(m_name2id.size());
  for (Name2ID::const_iterator it = m_name2id.begin(); it != m_name2id.end(); ++ it) {
    names.push_back(it->first);
  }
  // check for pv names that may have come directly through the store routine:
  for (TimeMap::const_iterator it = m_timeMap.begin(); it != m_timeMap.end(); ++ it) {
    bool nameAlreadyAdded = (m_name2id.find(it->first) != m_name2id.end());
    if (not nameAlreadyAdded) {
      names.push_back(it->first);
    }
  }
}

/// Get the list of PV aliases.
void 
EpicsStoreImpl::aliases(std::vector<std::string>& names) const
{
  names.clear();
  names.reserve(m_alias2id.size());
  for (Name2ID::const_iterator it = m_alias2id.begin(); it != m_alias2id.end(); ++ it) {
    names.push_back(it->first);
  }
}

/// Get alias name for specified PV name.
std::string
EpicsStoreImpl::alias(const std::string& pv) const
{
  std::string name;

  Name2ID::const_iterator it = m_name2id.find(pv);
  if (it != m_name2id.end()) {
    ID2Name::const_iterator ait = m_id2alias.find(it->second);
    if (ait != m_id2alias.end()) name = ait->second;
  }

  return name;
}

/// Get PV name for specified alias name.
std::string
EpicsStoreImpl::pvName(const std::string& alias) const
{
  std::string name;

  Name2ID::const_iterator ait = m_alias2id.find(alias);
  if (ait != m_alias2id.end()) {
    ID2Name::const_iterator it = m_id2name.find(ait->second);
    if (it != m_id2name.end()) name = it->second;
  }

  return name;
}


/// Get base class object for given EPICS PV name
boost::shared_ptr<Psana::Epics::EpicsPvHeader>
EpicsStoreImpl::getAny(const std::string& name) const
{
  // check for alias
  std::string pvName = this->pvName(name);
  if (pvName.empty()) pvName = name;

  // try TIME objects first
  TimeMap::const_iterator time_it = m_timeMap.find(pvName);
  if (time_it != m_timeMap.end()) return (time_it->second).first;

  // try CTRL objects
  CrtlMap::const_iterator ctrl_it = m_ctrlMap.find(pvName);
  if (ctrl_it != m_ctrlMap.end()) return ctrl_it->second;

  return boost::shared_ptr<Psana::Epics::EpicsPvHeader>();
}


//   Get status info for the EPICS PV.
void
EpicsStoreImpl::getStatus(const std::string& name, int& status, int& severity, PSTime::Time& time) const 
{
  // check for alias
  std::string pvName = this->pvName(name);
  if (pvName.empty()) pvName = name;

  // try TIME objects first
  TimeMap::const_iterator time_it = m_timeMap.find(pvName);
  if (time_it != m_timeMap.end()) {
    Psana::Epics::EpicsPvTimeHeader* tpv = (time_it->second).first.get();
    status = tpv->status();
    severity = tpv->severity();
    const Psana::Epics::epicsTimeStamp& stamp = tpv->stamp();
    time = PSTime::Time(stamp.sec()+sec_1970_to_1990, stamp.nsec());
    return;
  }

  // try CTRL objects
  CrtlMap::const_iterator ctrl_it = m_ctrlMap.find(pvName);
  if (ctrl_it != m_ctrlMap.end()) {
    Psana::Epics::EpicsPvCtrlHeader* cpv = ctrl_it->second.get();
    status = cpv->status();
    severity = cpv->severity();
    time = PSTime::Time();
    return;
  }

  // impossible to be here
  throw ExceptionEpicsName(ERR_LOC, name);
}

// Implementation of the getCtrl which returns generic pointer
boost::shared_ptr<Psana::Epics::EpicsPvCtrlHeader>
EpicsStoreImpl::getCtrlImpl(const std::string& name) const
{
  // check for alias
  std::string pvName = this->pvName(name);
  if (pvName.empty()) pvName = name;

  CrtlMap::const_iterator pvit = m_ctrlMap.find(pvName);
  if (pvit == m_ctrlMap.end()) return boost::shared_ptr<Psana::Epics::EpicsPvCtrlHeader>();
  return pvit->second;
}

// Implementation of the getTime which returns generic pointer
boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader>
EpicsStoreImpl::getTimeImpl(const std::string& name) const
{
  // check for alias
  std::string pvName = this->pvName(name);
  if (pvName.empty()) pvName = name;

  TimeMap::const_iterator pvit = m_timeMap.find(pvName);
  if (pvit == m_timeMap.end()) return boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader>();
  return (pvit->second).first;
}

} // namespace PSEnv
