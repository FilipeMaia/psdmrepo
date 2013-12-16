#ifndef TRANSLATOR_DONOTRANSLATE_H
#define TRANSLATOR_DONOTRANSLATE_H

#include <set>
#include "PSEvt/Event.h"


namespace Translator {

const std::string EMPTY_STRING = std::string();

/**
   @ingroup Translator

   @brief causes a Psana event to not be translated 

   skips translation of event data. An optional message log message can be 
   passed as well. The Translator will
   add this to a filterLog dataset off of the current CalibCycle group.
   
   It inserts a special class into the Event that the Translator.H5Output module
   will pick up, if the Translator.H5Output module runs after the module that calls 
   this function.

 */
void doNotTranslateEvent(PSEvt::Event &, const std::string &filterMsg=EMPTY_STRING);

class ExcludeEvent {
 public:
  ExcludeEvent() {};
  ExcludeEvent(std::string filterMsg) : m_filterMsg(filterMsg) {};
  const std::string & getMsg() { return m_filterMsg; };
  void setMsg(const std::string &newMsg) { m_filterMsg = newMsg; };
 private:
  std::string m_filterMsg;
};

} // namespace

#endif
