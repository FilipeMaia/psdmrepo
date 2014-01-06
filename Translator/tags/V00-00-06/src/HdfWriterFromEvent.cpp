#include "Translator/HdfWriterFromEvent.h"

namespace Translator {

std::ostream & operator<<(std::ostream &o, DataTypeLoc &dataTypeLoc) {
  switch (dataTypeLoc) {
  case inEvent:
    o << "inEvent";
    break;
  case inConfigStore:
    o << "inConfigStore";
    break;
    default:
      o<< "*error*";
      break;
  }
  return o;
} 

}
