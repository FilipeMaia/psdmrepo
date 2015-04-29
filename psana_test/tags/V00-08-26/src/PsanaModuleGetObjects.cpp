#include <stdexcept>
#include "psana/Module.h"

/**
 * Testing module that retrieves all objects from the EventStore. 
 * This is for testing small data.
 */
class PsanaModuleGetObjects : public Module {
public:
  PsanaModuleGetObjects(const std::string &name) : Module(name) {}
  void beginJob(Event &evt, Env &env) {
  };
  void event(Event &evt, Env &env) {
    boost::shared_ptr<ProxyDictI> proxyDict = evt.proxyDict();
    std::list<PSEvt::EventKey> keys = evt.keys();
    for (std::list<PSEvt::EventKey>::iterator pos = keys.begin(); pos != keys.end(); ++pos) {
      boost::shared_ptr<void> obj = proxyDict->get(pos->typeinfo(), PSEvt::Source(pos->src()), pos->key(), 0);
    }
  };
};

PSANA_MODULE_FACTORY(PsanaModuleGetObjects);
