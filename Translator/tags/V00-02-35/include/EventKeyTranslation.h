#ifndef TRANSLATOR_EVENTKEYTRANSLATION_H
#define TRANSLATOR_EVENTKEYTRANSLATION_H

#include <iostream>

#include "pdsdata/xtc/Damage.hh"
#include "PSEvt/EventKey.h"
#include "Translator/HdfWriterFromEvent.h"
#include "Translator/TypeClassEnum.h"

namespace Translator {

/**
 *  @ingroup Translator
 *
 *  @brief Describes how to translate eventKeys.
 *
 *  This struct is used to hold some information about how to translate
 *  EventKeys.  It holds the event damage, if a blank should be written, 
 *  and where the eventKey is in the configStore or eventStore.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
struct EventKeyTranslation {

  const PSEvt::EventKey eventKey;
  Pds::Damage damage;
  boost::shared_ptr<Translator::HdfWriterFromEvent> hdfWriter;
  typedef enum {UnknownEntry, NonBlank, Blank} EntryType;
  EntryType entryType;
  DataTypeLoc dataTypeLoc;
  TypeClass typeClass;
EventKeyTranslation() : damage(0), entryType(UnknownEntry), typeClass(UnknownType) {};
  EventKeyTranslation(const PSEvt::EventKey _eventKey, Pds::Damage _damage, 
                      boost::shared_ptr<HdfWriterFromEvent> _hdfWriter, 
                      EntryType _entryType, DataTypeLoc _dataTypeLoc, TypeClass _typeClass) :
  eventKey(_eventKey), damage(_damage), hdfWriter(_hdfWriter), 
    entryType(_entryType), dataTypeLoc(_dataTypeLoc), typeClass(_typeClass) {}
};

std::ostream & operator<<(std::ostream &o, EventKeyTranslation::EntryType entryType);

std::ostream & operator<<(std::ostream &o, EventKeyTranslation &ekt);

} // namespace

#endif
