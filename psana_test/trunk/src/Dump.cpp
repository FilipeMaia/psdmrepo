#include <sstream>

#include "PsanaTools/Dump.h"
#include "PsanaTools/psddl_Dump.h"

using namespace std;

namespace PsanaTools {

Dump::Dump(std::string name) : Module(name) {}
 
Dump::~Dump() {}

void Dump::beginJob(Event& evt, Env& env) 
{
}

void Dump::beginRun(Event& evt, Env& env) 
{
}

void Dump::beginCalibCycle(Event& evt, Env& env) 
{
}

void Dump::event(Event& evt, Env& env)
{
  list<EventKey> eventKeys = evt.keys();
  list<EventKey>::iterator keyIter;
  for (keyIter = eventKeys.begin(); keyIter != eventKeys.end(); ++keyIter) {
    EventKey &eventKey = *keyIter;
    PsanaTools::getAndDumpPsddlObject(evt, env, eventKey, true);
  }
}

void Dump::endCalibCycle(Event& evt, Env& env)
{
}

void Dump::endRun(Event& evt, Env& env)
{
}

void Dump::endJob(Event& evt, Env& env)
{
}


PSANA_MODULE_FACTORY(Dump);

}; // namespace PsanaTools
