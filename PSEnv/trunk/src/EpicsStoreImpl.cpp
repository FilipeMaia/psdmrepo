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
EpicsStoreImpl::store(const boost::shared_ptr<Psana::Epics::EpicsPvHeader>& pv, const Pds::Src& src)
{
  PvId pvid(src.log(), src.phy(), pv->pvId());

  if (pv->isTime()) {

    // find a name, or build fictional one
    std::string name;
    ID2Name::const_iterator it = m_id2name.find(pvid);
    if (it != m_id2name.end()) {
      name = it->second;
    } else {
      name = "PV:pvId=" + boost::lexical_cast<std::string>(pv->pvId()) +
          ":src_log=" + boost::lexical_cast<std::string>(src.log()) +
          ":src_phy=" + boost::lexical_cast<std::string>(src.phy());
      m_id2name.insert(std::make_pair(pvid, name));
    }

    MsgLog(logger, debug, "EpicsStore::store - storing TIME PV with id=" << pv->pvId());
    boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader> tpv =
        boost::static_pointer_cast<Psana::Epics::EpicsPvTimeHeader>(pv);
    m_timeMap[name] = tpv;

  } else if (pv->isCtrl()) {

    MsgLog(logger, debug, "EpicsStore::store - storing CTRL PV with id=" << pv->pvId());
    boost::shared_ptr<Psana::Epics::EpicsPvCtrlHeader> ctrl =
        boost::static_pointer_cast<Psana::Epics::EpicsPvCtrlHeader>(pv);
    std::string name = ctrl->pvName();
    m_id2name.insert(std::make_pair(pvid, name));
    m_ctrlMap[name] = ctrl;
    
  } else {
    
    MsgLog(logger, warning, "EpicsStore::store - unexpected PV type: ID=" << pv->pvId() << " type=" << pv->dbrType());
    
  }
}


/// Store alias name for EPICS PV.
void
EpicsStoreImpl::storeAlias(const Pds::Src& src, int pvId, const std::string& alias)
{
  m_alias2id[alias] = PvId(src.log(), src.phy(), pvId);
}


/// Get base class object for given EPICS PV name
boost::shared_ptr<Psana::Epics::EpicsPvHeader> 
EpicsStoreImpl::getAny(const std::string& name) const 
{
  // check for alias
  std::string pvName = alias2pv(name);
  if (pvName.empty()) pvName = name;

  // try TIME objects first
  TimeMap::const_iterator time_it = m_timeMap.find(pvName);
  if (time_it != m_timeMap.end()) return time_it->second;

  // try CTRL objects
  CrtlMap::const_iterator ctrl_it = m_ctrlMap.find(pvName);
  if (ctrl_it != m_ctrlMap.end()) return ctrl_it->second;
  
  return boost::shared_ptr<Psana::Epics::EpicsPvHeader>();
}

/// Get the list of PV names
void 
EpicsStoreImpl::pvNames(std::vector<std::string>& pvNames) const 
{
  pvNames.clear();
  pvNames.reserve(m_ctrlMap.size() + m_alias2id.size());
  for (ID2Name::const_iterator it = m_id2name.begin(); it != m_id2name.end(); ++ it) {
    pvNames.push_back(it->second);
  }
  for (Alias2ID::const_iterator it = m_alias2id.begin(); it != m_alias2id.end(); ++ it) {
    pvNames.push_back(it->first);
  }
}

// Implementation of the getCtrl which returns generic pointer
boost::shared_ptr<Psana::Epics::EpicsPvCtrlHeader> 
EpicsStoreImpl::getCtrlImpl(const std::string& name) const
{
  // check for alias
  std::string pvName = alias2pv(name);
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
  std::string pvName = alias2pv(name);
  if (pvName.empty()) pvName = name;

  TimeMap::const_iterator pvit = m_timeMap.find(pvName);
  if (pvit == m_timeMap.end()) return boost::shared_ptr<Psana::Epics::EpicsPvTimeHeader>();
  return pvit->second;
}

//   Get status info for the EPICS PV.
void
EpicsStoreImpl::getStatus(const std::string& name, int& status, int& severity, PSTime::Time& time) const 
{
  // check for alias
  std::string pvName = alias2pv(name);
  if (pvName.empty()) pvName = name;

  // try TIME objects first
  TimeMap::const_iterator time_it = m_timeMap.find(pvName);
  if (time_it != m_timeMap.end()) {
    Psana::Epics::EpicsPvTimeHeader* tpv = time_it->second.get();
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

/// return PV name for given alias name or empty string if alias not defined
std::string
EpicsStoreImpl::alias2pv(const std::string& name) const
{
  Alias2ID::const_iterator alias_it = m_alias2id.find(name);
  if (alias_it != m_alias2id.end()) {
    // PV name
    ID2Name::const_iterator id_it = m_id2name.find(alias_it->second);
    if (id_it != m_id2name.end()) return id_it->second;
  }

  return std::string();
}

} // namespace PSEnv
