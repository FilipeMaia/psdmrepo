//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpDamage
//
// Author List:
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpDamage.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <list>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpDamage)

namespace {
  
  struct LocationAndDamage {
    bool inCfg;
    bool inCalib;
    bool inEvent;
    bool inDamageMap;
    Pds::Damage damage;
    std::string eventKeyString;

    LocationAndDamage() : inCfg(false), inCalib(false), inEvent(false), inDamageMap(false), damage(0) {};

    LocationAndDamage(bool _inCfg, bool _inCalib, bool _inEvent, bool _inDamageMap, 
                      Pds::Damage _damage, std::string _eventKeyString) 
      : inCfg(_inCfg), inCalib(_inCalib), inEvent(_inEvent), 
        inDamageMap(_inDamageMap), damage(_damage), eventKeyString(_eventKeyString) {}

    LocationAndDamage(const LocationAndDamage &other) 
      : inCfg(other.inCfg), inCalib(other.inCalib), inEvent(other.inEvent), 
        inDamageMap(other.inDamageMap), damage(other.damage), eventKeyString(other.eventKeyString) {}

    LocationAndDamage & operator=(const LocationAndDamage &other) {
      inCfg = other.inCfg;
      inCalib = other.inCalib;
      inEvent = other.inEvent;
      inDamageMap = other.inDamageMap;
      damage = other.damage;
      eventKeyString = other.eventKeyString;
      return *this;
    }

    bool operator <(const struct LocationAndDamage &other) const {
      if (inCfg < other.inCfg) return true;
      if (inCfg > other.inCfg) return false;
      if (inCalib < other.inCalib) return true;
      if (inCalib > other.inCalib) return false;
      if (inEvent < other.inEvent) return true;
      if (inEvent > other.inEvent) return false;
      if (inDamageMap < other.inDamageMap) return true;
      if (inDamageMap > other.inDamageMap) return false;
      if (damage.value() < other.damage.value()) return true;
      if (damage.value() > other.damage.value()) return false;
      return (eventKeyString < other.eventKeyString);
    }
  };

  void
  setDamage(const boost::shared_ptr<PSEvt::DamageMap> damageMap, const PSEvt::EventKey & key, 
            bool &inDamageMap, Pds::Damage & damage) 
  {
    inDamageMap = false;
    damage = Pds::Damage(0);
    if (damageMap) {
      PSEvt::DamageMap::const_iterator damagePos = damageMap->find(key);
      if (damagePos != damageMap->end()) {
        inDamageMap = true;
        damage = damagePos->second;
      }
    }
  }
  

} // local namespace

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpDamage::DumpDamage (const std::string& name)
  : Module(name)
{
  m_totalEvents = -1;
}

//--------------
// Destructor --
//--------------
DumpDamage::~DumpDamage ()
{
}

void 
DumpDamage::printKeysAndDamage(std::ostream& out, Event &evt, Env &env) {

  m_damageMap = evt.get();
  bool inDamageMap;
  Pds::Damage damage;

  std::map<PSEvt::EventKey, LocationAndDamage> allKeys;
  std::list<EventKey>::const_iterator iter;

  const PSEvt::HistI * configHist  = env.configStore().proxyDict()->hist();
  if (not configHist) MsgLog(name(),fatal,"Internal error - no HistI object in configStore");
  
  const std::list<EventKey> configKeys = env.configStore().keys();

  // gather list of event keys and their damage for printing

  // find config keys that we have not seen or that have changed since we last saw them
  for (iter = configKeys.begin(); iter != configKeys.end(); ++iter) {
    const EventKey &configKey = *iter;
    if ( (m_configUpdates.find(configKey) == m_configUpdates.end()) or
         (m_configUpdates[configKey] < configHist->updates(configKey)) ) {
      setDamage(m_damageMap, configKey, inDamageMap, damage);
      std::ostringstream configKeyString;
      configKeyString << configKey;
      allKeys[configKey] = LocationAndDamage(true,false,false,inDamageMap,damage,configKeyString.str());
      m_configUpdates[configKey] = configHist->updates(configKey);
    }
  }

  const PSEvt::HistI * calibHist  = env.calibStore().proxyDict()->hist();
  if (not calibHist) MsgLog(name(),fatal,"Internal error - no HistI object in calibStore");

  const std::list<EventKey> calibKeys = env.calibStore().keys();

  // find calib keys that we have not seen or that have changed since we last saw them
  for (iter = calibKeys.begin(); iter != calibKeys.end(); ++iter) {
    const EventKey &calibKey = *iter;
    if ( (m_calibUpdates.find(calibKey) == m_calibUpdates.end()) or
         (m_calibUpdates[calibKey] < calibHist->updates(calibKey)) ) {
      setDamage(m_damageMap, calibKey, inDamageMap, damage);
      std::ostringstream calibKeyString;
      calibKeyString << calibKey;
      std::map<PSEvt::EventKey, LocationAndDamage>::iterator keyPos = allKeys.find(calibKey);
      if (keyPos != allKeys.end()) {
        // a key in both the calib and config store seems odd, print warning
        MsgLog(name(),warning," calib key: " << calibKey << " was also in configStore");
        keyPos->second.inCalib = true;
        if ((keyPos->second.inDamageMap != inDamageMap) or
            (keyPos->second.damage.value() != damage.value())) {
          // Getting different damage values for the same key suggests a bug
          MsgLog(name(),error,"  calib key " << calibKey << " was also in configStore " 
                 << " with different damage.  Discarding new calib damage values which are: "
                 << "found in damageMap=" << inDamageMap
                 << " damage= 0x" << std::hex << damage.value());
        }
      } else {
        allKeys[calibKey] = LocationAndDamage(false,true,false,inDamageMap,damage,calibKeyString.str());
      }
      m_calibUpdates[calibKey] = calibHist->updates(calibKey);
    }
  }

  const std::list<EventKey> eventKeys = evt.keys();

  // now go through all even keys
  for (iter = eventKeys.begin(); iter != eventKeys.end(); ++iter) {
    const EventKey &eventKey = *iter;
    setDamage(m_damageMap, eventKey, inDamageMap, damage);
    std::ostringstream eventKeyString;
    eventKeyString << eventKey;
    std::map<PSEvt::EventKey, LocationAndDamage>::iterator keyPos = allKeys.find(eventKey);
    if (keyPos == allKeys.end()) {
      allKeys[eventKey] = LocationAndDamage(false,false,true,inDamageMap,damage, eventKeyString.str());
    } else {
      LocationAndDamage &locAndDamage = keyPos->second;
      locAndDamage.inEvent = true;
      if ((locAndDamage.inDamageMap != inDamageMap) or (locAndDamage.damage.value() != damage.value())) {
        // Getting different damage values for the same key suggests a bug
        MsgLog(name(), error, "event key " << eventKey 
               << " was also a config or calib keys but with different damage. old values: "
               << "inDamageMap= " 
               << locAndDamage.inDamageMap  << " damage= 0x" 
               << std::hex << locAndDamage.damage.value()
               << " discarding old values");
        locAndDamage.inDamageMap = inDamageMap;
        locAndDamage.damage = damage;
      }
    }
  }

  // find all keys that are damaged, but were not in the event
  if (m_damageMap) {
    PSEvt::DamageMap::const_iterator damageIter;
    for (damageIter = m_damageMap->begin(); damageIter != m_damageMap->end(); ++damageIter) {
      const PSEvt::EventKey &eventKey = damageIter->first;
      Pds::Damage damage = damageIter->second;
      std::ostringstream eventKeyString;
      eventKeyString << eventKey;
      std::map<PSEvt::EventKey, LocationAndDamage>::iterator keyPos = allKeys.find(eventKey);
      if (keyPos == allKeys.end()) {
        allKeys[eventKey] = LocationAndDamage(false,false,false,true,damage, eventKeyString.str());
      }
    }
  }
      
  std::vector<LocationAndDamage> allKeysList;
  std::map<PSEvt::EventKey, LocationAndDamage>::iterator pos;
  for (pos = allKeys.begin(); pos != allKeys.end(); ++pos) {
    allKeysList.push_back(pos->second);
  }
  sort(allKeysList.begin(), allKeysList.end());

  // finally, print all the keys
  std::vector<LocationAndDamage>::iterator printPos;
  for (printPos = allKeysList.begin(); printPos != allKeysList.end(); ++printPos) {
    LocationAndDamage &locAndDamage = *printPos;
    out << "    ";
    out << ((locAndDamage.inCfg) ? "cfg" : "---");
    out << ((locAndDamage.inCalib) ? " clb" : " ---");
    out << ((locAndDamage.inEvent) ? " evt" : " ---");
    out << ((locAndDamage.inDamageMap) ? " dmg" : " ---");
    out << " ";
    std::ostringstream damageBits;
    if (locAndDamage.inDamageMap) {
      out << "0x" << std::setw(8) << std::setfill('0') << std::hex << locAndDamage.damage.value();
      uint32_t value = locAndDamage.damage.value();
      if (value) {
        bool DroppedContribution    = value & (1<<Pds::Damage::DroppedContribution);
        bool Uninitialized          = value & (1<<Pds::Damage::Uninitialized);
        bool OutOfOrder             = value & (1<<Pds::Damage::OutOfOrder);
        bool OutOfSynch             = value & (1<<Pds::Damage::OutOfSynch);
        bool UserDefined            = value & (1<<Pds::Damage::UserDefined);
        bool IncompleteContribution = value & (1<<Pds::Damage::IncompleteContribution);
        bool ContainsIncomplete     = value & (1<<Pds::Damage::ContainsIncomplete);
        damageBits << "dropped=" << DroppedContribution;
        damageBits << " uninitialized="<<Uninitialized;
        damageBits << " OutOfOrder="<<OutOfOrder;
        damageBits << " OutOfSynch="<<OutOfSynch;
        damageBits << " UserDefined="<<UserDefined;
        damageBits << " IncompleteContribution="<<IncompleteContribution;
        damageBits << " ContainsIncomplete="<<ContainsIncomplete;
        damageBits << " userBits=0x"<< locAndDamage.damage.userBits();
      }
    } else {
      out << "          ";
    }
    out << " " << locAndDamage.eventKeyString << '\n';
    std::string damageBitsStr = damageBits.str();
    if (damageBitsStr.size()) {
      out << "                    " << damageBitsStr << '\n';
    }
  }
  // report on any src only damage:
  if (m_damageMap) {
    const std::vector<std::pair<Pds::Src,Pds::Damage> > & srcDmgList = 
      m_damageMap->getSrcDroppedContributions();
    if (srcDmgList.size()>0) {
      out << " -------- src damage with dropped contribs --------- \n";
      for (unsigned idx=0; idx < srcDmgList.size(); ++idx) {
        const std::pair<Pds::Src,Pds::Damage> &srcDmgPair = srcDmgList.at(idx);
        const Pds::Src &src = srcDmgPair.first;
        const Pds::Damage &dmg = srcDmgPair.second;
        out << "   0x" << std::setw(8) << std::setfill('0') << std::hex << dmg.value();
        out << "  " << src << '\n';
      }
    }
  }
}


void 
DumpDamage::beginJob(Event& evt, Env& env)
{
  m_runNumber = -1;
  MsgLog(name(), info, " beginJob()");
  printKeysAndDamage(std::cout, evt, env);
}

void 
DumpDamage::beginRun(Event& evt, Env& env)
{
  m_calibCycleNumber = -1;
  MsgLog(name(),info," beginRun(): run=" << ++m_runNumber);
  printKeysAndDamage(std::cout, evt, env);
}

void 
DumpDamage::beginCalibCycle(Event& evt, Env& env)
{
  m_eventNumber = -1;
  MsgLog(name(),info," beginCalibCycle() run=" << m_runNumber << " calib=" << ++m_calibCycleNumber);
  printKeysAndDamage(std::cout, evt, env);
}

void 
DumpDamage::event(Event& evt, Env& env)
{
  MsgLog(name(),info," event() run=" << m_runNumber 
         << " calib=" << m_calibCycleNumber << " eventNumber=" 
         << ++m_eventNumber << " totalEvents= " << ++m_totalEvents );
  printKeysAndDamage(std::cout, evt, env);
}

/// Method which is called at the end of the calibration cycle
void
DumpDamage::endCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), info, " endCalibCycle() run=" << m_runNumber << " calib=" << m_calibCycleNumber);
  printKeysAndDamage(std::cout, evt, env);
}

/// Method which is called at the end of the run
void
DumpDamage::endRun(Event& evt, Env& env)
{
  MsgLog(name(), info, " endRun() run=" << m_runNumber);
  printKeysAndDamage(std::cout, evt, env);
}

/// Method which is called once at the end of the job
void
DumpDamage::endJob(Event& evt, Env& env)
{
  MsgLog(name(), info, " endJob()");
  printKeysAndDamage(std::cout, evt, env);
}


} // namespace psana_examples
