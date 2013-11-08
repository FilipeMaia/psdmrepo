#ifndef TRANSLATOR_EVENTKEYTRANSLATION_H
#define TRANSLATOR_EVENTKEYTRANSLATION_H

#include <iostream>

#include "pdsdata/xtc/Damage.hh"
#include "PSEvt/EventKey.h"
#include "Translator/HdfWriterBase.h"

namespace Translator {

  struct EventKeyTranslation {
    const PSEvt::EventKey eventKey;
    Pds::Damage damage;
    boost::shared_ptr<Translator::HdfWriterBase> hdfWriter;
    typedef enum { NonBlank, Blank} EntryType;
    EntryType entryType;
    DataTypeLoc dataTypeLoc;
    EventKeyTranslation() {};
    EventKeyTranslation(const EventKey _eventKey, Pds::Damage _damage, 
                        boost::shared_ptr<HdfWriterBase> _hdfWriter, 
                        EntryType _entryType, DataTypeLoc _dataTypeLoc) :
    eventKey(_eventKey), damage(_damage), hdfWriter(_hdfWriter), 
      entryType(_entryType), dataTypeLoc(_dataTypeLoc) {}
  };

  std::ostream & operator<<(std::ostream &o, EventKeyTranslation::EntryType entryType) {
    switch (entryType) {
    case EventKeyTranslation::NonBlank:
      o << "NonBlank";
      break;
    case EventKeyTranslation::Blank:
      o << "Blank";
      break;
    default:
      o << "*error*";
    }
    return o;
  }

  std::ostream & operator<<(std::ostream &o, EventKeyTranslation &ekt) {
    o.setf(std::ios::hex,std::ios::basefield);
    o << "eventKey: " << ekt.eventKey
      << " damage: " << ekt.damage.value() 
      << " hdfWriter: " << ekt.hdfWriter << " entryType: " << ekt.entryType
      << " dataTypeLoc: " << ekt.dataTypeLoc;
      o.unsetf(std::ios::hex);
      return o;
  }

} // namespace

#endif
