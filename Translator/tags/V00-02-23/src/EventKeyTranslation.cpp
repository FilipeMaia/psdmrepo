#include "Translator/EventKeyTranslation.h"

using namespace Translator;

std::ostream & Translator::operator<<(std::ostream &o, EventKeyTranslation::EntryType entryType) {
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

std::ostream & Translator::operator<<(std::ostream &o, EventKeyTranslation &ekt) {
  o.setf(std::ios::hex,std::ios::basefield);
  o << "eventKey: " << ekt.eventKey
    << " damage: " << ekt.damage.value() 
    << " hdfWriter: " << ekt.hdfWriter << " entryType: " << ekt.entryType
    << " dataTypeLoc: " << ekt.dataTypeLoc;
  o.unsetf(std::ios::hex);
  return o;
}
