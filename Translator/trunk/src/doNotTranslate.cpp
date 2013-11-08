#include <string>

#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"

#include "Translator/doNotTranslate.h"
#include "MsgLogger/MsgLogger.h"

using namespace Translator;

namespace {
  char * mainlogger = "doNoTranslate";
}

void Translator::doNotTranslateEvent(PSEvt::Event &evt, const std::string & filterMsg) {
  // see if another module already added an ExcludeEvent:
  boost::shared_ptr<ExcludeEvent> excludeEvent = evt.get();
  if (excludeEvent) {
    MsgLog(mainlogger,warning,"Filtering previously filtered event. discarding previous msg: " << excludeEvent->getMsg());
    excludeEvent->setMsg(filterMsg);
    return;
  }
  excludeEvent = boost::make_shared<ExcludeEvent>(filterMsg);
  evt.put(excludeEvent);
}

////////////////////
// below is for testing - we 
// make a module that will filter out events.
// The module - TestDoNotTranslate 
//  takes the two configuration parameters:
//    skip = 0 1 2
//    messages = message0  message1  message2
//  It uses 0-up counter for the events, the conter indexes all events over
//  all calib cycles.  
//  The number of entries (space separated) in skip must equal the number of entries in messages.

#include <vector>
#include "psana/Module.h"

namespace {
  char * testlogger = "doNoTranslate";
}

namespace Translator {
  
class TestDoNotTranslate : public Module {
public:
  TestDoNotTranslate(std::string moduleName) : Module(moduleName) {
    m_eventsToSkip = configList("skip");
    m_messages = configList("messages");
    if (m_eventsToSkip.size() != m_messages.size()) {
      MsgLog(testlogger,fatal,"number of events to skip: " << m_eventsToSkip.size() << " not equal to number of messages " << m_messages.size());
    }
    WithMsgLog(testlogger,trace,str) {
      str << "eventsToSkip: ";
      for (size_t idx=0; idx < m_eventsToSkip.size(); ++idx) str << m_eventsToSkip.at(idx) << " ";
      str << std::endl << "messages: ";
      for (size_t idx=0; idx < m_messages.size(); ++idx) str <<m_messages.at(idx) << " ";
    }
  }
  virtual void beginJob(Event& evt, Env& env) {
    m_eventCounter = 0;
  }
  virtual void event(Event& evt, Env& env) {
    for (size_t idx=0; idx < m_eventsToSkip.size(); ++idx) {
      if (m_eventsToSkip.at(idx) == m_eventCounter) {
        doNotTranslateEvent(evt, m_messages.at(idx));
        doNotTranslateEvent(evt, m_messages.at(idx));
      }
    }
    ++m_eventCounter;
  }
  private:
  std::vector<size_t> m_eventsToSkip;
  std::vector<std::string> m_messages;
  size_t m_eventCounter;
};

PSANA_MODULE_FACTORY(TestDoNotTranslate);

}
