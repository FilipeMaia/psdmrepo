namespace PSEvt {
  class Event;
  class EventKey;
};

namespace PSEnv {
  class Env;
};

namespace PsanaTools {

void getAndDumpPsddlObject(PSEvt::Event &evt, PSEnv::Env &env, PSEvt::EventKey &eventKey, bool inEvt);


} // namespace PsanaTools
