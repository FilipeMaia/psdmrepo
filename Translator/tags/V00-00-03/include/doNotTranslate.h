#ifndef TRANSLATOR_DONOTRANSLATE_H
#define TRANSLATOR_DONOTRANSLATE_H

#include <set>
#include "PSEvt/Event.h"

/**
 Call the function doNotTranslateEvent to skip translation of event data.
 An optional message log message can be passed as well. The Translator will
 add this to a filterLog dataset off of the current CalibCycle group.
 
  It inserts a special class into the Event that the Translator will pick up
 (Assuming the Translator.H5output module is listed after the calling module
 in the Psana module list).

 */

namespace Translator {

const std::string EMPTY_STRING = std::string();

void doNotTranslateEvent(PSEvt::Event &, const std::string &filterMsg=EMPTY_STRING);

class ExcludeEvent {
 public:
  ExcludeEvent() {};
  ExcludeEvent(std::string filterMsg) : m_filterMsg(filterMsg) {};
  const std::string & getMsg() { return m_filterMsg; }; // TODO: is this dangerous?  return reference to 
                                                        // internal data, boost smart pointers don't know about
                                                        // this piece
  void setMsg(const std::string &newMsg) { m_filterMsg = newMsg; };
 private:
  std::string m_filterMsg;
};

} // namespace

#endif
