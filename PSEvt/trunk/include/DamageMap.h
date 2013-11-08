#ifndef PSEVT_DAMAGEMAP_H
#define PSEVT_DAMAGEMAP_H

#include <map>
#include <iostream>
#include <vector>
#include <utility>

#include "pdsdata/xtc/Damage.hh"
#include "PSEvt/EventKey.h"

namespace PSEvt {

/**
 *  @ingroup PSEvt
 *  
 *  @brief Class to map EventKeys to their xtc damage values, and to hold Src only damage
 *  
 *  If a Psana Input Module records damage during event processing, for objects placed 
 *  in either the EventStore or ConfigStore, the input module should place an instance of 
 *  DamageMap in the Event.  Any damage to objects placed into the Event or ConfigStore 
 *  should be added to the DamageMap with the same EventKey used to store the object in the 
 *  Event or ConfigStore.  Damage associated only with a Pds::Src value should also be placed
 *  in the DamageMap. Src damage should always contain the DroppedContribution bit.  If it
 *  is not present, the damage value is ignored.
 *
 *  Modules can then look for the DamageMap in the Event and find out if there was any
 *  damage for specific EventKey's, as well as find out if dropped contributions (src damage)
 *  were present. This kind of damage implies a split event.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see EventKey
 *
 *  @version \$Id:
 *
 *  @author David Schneider
 */

class DamageMap {
public:
  typedef std::map<PSEvt::EventKey, Pds::Damage> MapImpl;
  typedef MapImpl::iterator iterator;
  typedef MapImpl::const_iterator const_iterator;

  /**
   * @brief add damage to DamageMap.  Works the same as std::map::operator[]
   */
  Pds::Damage & operator[](const PSEvt::EventKey &eventKey) { return m_map[eventKey]; }

  iterator begin() { return m_map.begin(); }
  iterator end() {return m_map.end(); }
  /**
   * @brief finds damage in the DamageMap. Tries two keys if eventKey has key string.
   *
   * EventKey's have three componenents, type, Src and key, where key is a string that
   * users may create, typically for data derived from original data in the Psana Event store.  
   * If no damage has been recorded for the given key, and the given key includes a string 
   * component, find() will also look for damage associated with the corresponding EventKey 
   * where the string component is blank.
   *
   * @param[in] eventKey  EventKey to look up damage for
   * @return iterator to eventKey position in map. 
   */
  iterator find(const PSEvt::EventKey &eventKey);

  const_iterator begin() const { return m_map.begin(); }
  const_iterator end() const { return m_map.end(); }
  /**
   * @brief const version of find(), see documentation for find().
   */
  const_iterator find(const PSEvt::EventKey &eventKey) const;
  
  void addSrcDamage(Pds::Src src, Pds::Damage damage);
  bool splitEvent() const { return m_droppedContribs.size()>0; }

  /**
   * @brief returns a histogram of damage values in the DamageMap associated with EventKeys 
   *
   * distinct Damage values are mapped to their frequency within the DamageMap.  No reporting
   * is provided for Src damage - use getSrcDroppedContributions or splitEvent for this
   */
  std::map<uint32_t, int> damageCounts() const;

  /**
   * @brief returns the list of src only damage for the event.  
   *
   * All damage values will contain the DroppedContribution bit.
   */
  const std::vector<std::pair<Pds::Src,Pds::Damage> > & getSrcDroppedContributions() const { return m_droppedContribs; }

 private:
  MapImpl m_map;
  std::vector<std::pair<Pds::Src,Pds::Damage> > m_droppedContribs;
  
};
   
std::ostream & operator<<(std::ostream &, const DamageMap &);

}; // namespace

#endif

