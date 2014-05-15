#include <string>

namespace PSEvt {
  class Event;
  class EventKey;
};

namespace PSEnv {
  class Env;
};

namespace psana_test {

void getAndDumpPsddlObject(PSEvt::Event &evt, PSEnv::Env &env, PSEvt::EventKey &eventKey, bool inEvt);

} // namespace psana_test
