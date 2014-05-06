#include <sstream>
#include <iostream>

#include "psana_test/Dump.h"
#include "psana_test/psddl_Dump.h"

#include "PSEvt/EventId.h"

using namespace std;

namespace psana_test {

Dump::Dump(std::string name) : Module(name) 
{}
 
Dump::~Dump() 
{}

void Dump::beginJob(Event& evt, Env& env) 
{
  cout <<  "==================" << endl;
  cout <<  "=== begin job ====" << endl;
  list<EventKey> eventKeys = env.configStore().keys();
  list<EventKey>::iterator keyIter;
  for (keyIter = eventKeys.begin(); keyIter != eventKeys.end(); ++keyIter) {
    EventKey &eventKey = *keyIter;
    //   psana_test::getAndDumpPsddlObject(evt, env, eventKey, false);
  }
}

void Dump::event(Event& evt, Env& env)
{
  boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
  PSTime::Time time = eventId->time();
  unsigned fid = eventId->fiducials();
  time_t sec = time.sec();
  uint32_t nsec = time.nsec();
  cout << "===============================================================";
  cout << "=== event: seconds= " << sec << " nanoseconds= " << nsec << " fiducials= " << fid << endl;
  list<EventKey> eventKeys = evt.keys();
  list<EventKey>::iterator keyIter;
  for (keyIter = eventKeys.begin(); keyIter != eventKeys.end(); ++keyIter) {
    EventKey &eventKey = *keyIter;
    //    psana_test::getAndDumpPsddlObject(evt, env, eventKey, true);
  }
}


PSANA_MODULE_FACTORY(Dump);

}; // namespace psana_test
