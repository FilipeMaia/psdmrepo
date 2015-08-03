#include <string>

#include "MsgLogger/MsgLogger.h"
#include "Translator/specialKeyStrings.h"

namespace {

const std::string do_not_translate("do_not_translate");
const std::string translate_vlen("translate_vlen");

bool hasPrefix(const std::string &prefix, const std::string &key, 
               std::string *keyWithPrefixStripped) 
{
  if (key.size() < prefix.size()) {
    if (keyWithPrefixStripped != NULL) {
      *keyWithPrefixStripped = key;
    }
    return false;
  }
  
  bool startsWithPrefix = true;
  unsigned idx=0;
  while ((idx < prefix.size()) and startsWithPrefix) {
    startsWithPrefix = (key[idx] == prefix[idx]);
    ++idx;
  }
  if (keyWithPrefixStripped != NULL) {
    if (not startsWithPrefix) *keyWithPrefixStripped = key;
    else {
      unsigned afterPrefix = prefix.size();
      if ((key.size() > afterPrefix) and (key[afterPrefix] == ':' or key[afterPrefix]=='_')) {
        afterPrefix++;
      }
      if (afterPrefix <= key.size()) *keyWithPrefixStripped = key.substr(afterPrefix);
      else *keyWithPrefixStripped = "";
    }
  }
  return startsWithPrefix;
}


} // local namespace

const std::string & Translator::doNotTranslatePrefix() { 
  return do_not_translate;
}

const std::string & Translator::ndarrayVlenPrefix() { 
  return translate_vlen;
}

bool Translator::hasDoNotTranslatePrefix(const std::string &key, 
                                         std::string *keyWithPrefixStripped) 
{
  return hasPrefix(doNotTranslatePrefix(), key, keyWithPrefixStripped);
}

bool Translator::hasNDArrayVlenPrefix(const std::string &key, 
                                      std::string *keyWithPrefixStripped) 
{
  return hasPrefix(ndarrayVlenPrefix(), key, keyWithPrefixStripped);
}


////////////////////
// below is for testing - we 
// make a module that will filter out events.
// The module - TestModuleDoNotTranslate 
//  takes the two configuration parameters:
//    skip = 0 1 2
//    messages = message0  message1  message2
//  It uses 0-up counter for the events, the conter indexes all events over
//  all calib cycles.  
//  The number of entries (space separated) in skip must equal the number of entries in messages.
//  For each of the events listed in skip, the module will call the
//  
//    doNotTranslate() function, which will tell the Translator.H5Output module to 
//    skip this event.
#include <vector>
#include <string>
#include "psana/Module.h"

namespace Translator {
  
class TestModuleDoNotTranslate : public Module {
public:
  TestModuleDoNotTranslate(std::string moduleName) : Module(moduleName) {
    m_eventsToSkip = configList("skip");
    m_messages = configList("messages");
    m_key = configStr("key","");
    m_doNotTranslateKey = "do_not_translate";
    if (m_key.size()>0) {
      m_doNotTranslateKey += ':';
      m_doNotTranslateKey += m_key;
    }
    if (m_eventsToSkip.size() != m_messages.size()) {
      MsgLog(name(),fatal,"number of events to skip: " << m_eventsToSkip.size() << " not equal to number of messages " << m_messages.size());
    }
    WithMsgLog(name(),trace,str) {
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
        boost::shared_ptr<std::string> message = boost::make_shared<std::string>(m_messages.at(idx));        
        evt.put(message,m_doNotTranslateKey);
      }
    }
    ++m_eventCounter;
  }
private:
  std::vector<size_t> m_eventsToSkip;
  std::vector<std::string> m_messages;
  size_t m_eventCounter;
  std::string m_key;
  std::string m_doNotTranslateKey;
};
  
PSANA_MODULE_FACTORY(TestModuleDoNotTranslate);

}
