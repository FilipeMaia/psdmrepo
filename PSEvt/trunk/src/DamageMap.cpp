#include "PSEvt/DamageMap.h"
#include "MsgLogger/MsgLogger.h"

namespace {
  const char * logger = "DamageMap";
}

PSEvt::DamageMap::iterator PSEvt::DamageMap::find(const PSEvt::EventKey &eventKey) {
  PSEvt::DamageMap::iterator pos = m_map.find(eventKey);
  if (pos == m_map.end()) {
    if (eventKey.key().size() > 0) {
      PSEvt::EventKey eventKeyNoStr(eventKey.typeinfo(), eventKey.src(),"");
      PSEvt::DamageMap::iterator posNoStr = m_map.find(eventKeyNoStr);
      return posNoStr;
    }
  }
  return pos;
}

PSEvt::DamageMap::const_iterator PSEvt::DamageMap::find(const PSEvt::EventKey &eventKey) const {
  PSEvt::DamageMap::const_iterator pos = m_map.find(eventKey);
  if (pos == m_map.end()) {
    if (eventKey.key().size() > 0) {
      PSEvt::EventKey eventKeyNoStr(eventKey.typeinfo(), eventKey.src(),"");
      PSEvt::DamageMap::const_iterator posNoStr = m_map.find(eventKeyNoStr);
      return posNoStr;
    }
  }
  return pos;
}

void PSEvt::DamageMap::addSrcDamage(Pds::Src src, Pds::Damage damage) {
  if ( damage.bits() & (1<<Pds::Damage::DroppedContribution)) {
    m_droppedContribs.push_back(std::make_pair(src,damage));
  } else {
    MsgLog(logger,trace,"Unexpected, addSrcDamage called but damage does not contain "
           << "DroppedContribution bit, src= " << src << " damage = " << damage.value());
  }
}

std::map<uint32_t,int> PSEvt::DamageMap::damageCounts() const {
  std::map<uint32_t,int> counts;
  for (const_iterator pos = begin(); pos != end(); ++pos) {
    uint32_t damageValue = pos->second.value();
    int &count = counts[damageValue];
    ++count;
  }
  return counts;
}

/// print everything in the DamageMap - EventKeys and damage as well as 
/// DroppedContribution damage
std::ostream & PSEvt::operator<<(std::ostream &o, const PSEvt::DamageMap &damageMap) {
  PSEvt::DamageMap::const_iterator pos;
  bool first = true;
  o.setf(std::ios::hex,std::ios::basefield);
  for (pos = damageMap.begin(); pos != damageMap.end(); ++pos) {
    const PSEvt::EventKey &eventKey = pos->first;
    const Pds::Damage & damage = pos->second;
    if (not first) o << ", ";
    first = false;
    o << eventKey << " damage=0x" << damage.value();
  }
  if (damageMap.splitEvent()) {
    const std::vector<std::pair<Pds::Src,Pds::Damage> > &droppedContribs = damageMap.getSrcDroppedContributions();
    o << "DroppedContributions: ";
    for (size_t idx=0; idx < droppedContribs.size(); ++idx) {
      const std::pair<Pds::Src,Pds::Damage> & srcDamagePair = droppedContribs.at(idx);
      const Pds::Src & src = srcDamagePair.first;
      const Pds::Damage & damage = srcDamagePair.second;
      o << "src="<< src << " dmg=0x" <<damage.value();
      if (idx < droppedContribs.size()-1) {
        o << ", ";
      }
    }
  }
  o.unsetf(std::ios::hex);
  return o;
}
