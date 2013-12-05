#ifndef TRANSLATOR_EVENTKEYTRANSLATION_H
#define TRANSLATOR_EVENTKEYTRANSLATION_H

#include <iostream>

#include "pdsdata/xtc/Damage.hh"
#include "PSEvt/EventKey.h"
#include "Translator/HdfWriterFromEvent.h"

namespace Translator {

struct EventKeyTranslation {
  const PSEvt::EventKey eventKey;
  Pds::Damage damage;
  boost::shared_ptr<Translator::HdfWriterFromEvent> hdfWriter;
  typedef enum {NonBlank, Blank} EntryType;
  EntryType entryType;
  DataTypeLoc dataTypeLoc;
  EventKeyTranslation() {};
  EventKeyTranslation(const PSEvt::EventKey _eventKey, Pds::Damage _damage, 
                      boost::shared_ptr<HdfWriterFromEvent> _hdfWriter, 
                      EntryType _entryType, DataTypeLoc _dataTypeLoc) :
  eventKey(_eventKey), damage(_damage), hdfWriter(_hdfWriter), 
    entryType(_entryType), dataTypeLoc(_dataTypeLoc) {}
};

std::ostream & operator<<(std::ostream &o, EventKeyTranslation::EntryType entryType);

std::ostream & operator<<(std::ostream &o, EventKeyTranslation &ekt);

} // namespace

#endif
