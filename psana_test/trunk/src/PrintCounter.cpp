#include <sstream>
#include <iostream>

#include "psana_test/PrintCounter.h"

using namespace std;

namespace psana_test {

PrintCounter::PrintCounter(std::string name) : Module(name),
                               m_run(-1), m_calibCycle(-1), m_event(-1)
{}
 
PrintCounter::~PrintCounter() {}

void PrintCounter::beginJob(Event& evt, Env& env) 
{
  m_run = -1;
}

void PrintCounter::beginRun(Event& evt, Env& env) 
{
  m_run += 1;
  m_calibCycle = -1;
}

void PrintCounter::beginCalibCycle(Event& evt, Env& env) 
{
  m_calibCycle += 1;
  m_event = -1;
}

void PrintCounter::event(Event& evt, Env& env)
{
  m_event += 1;
  cout << "run=" << m_run << " calibcycle=" 
       << m_calibCycle << " event=" << m_event << endl;
}

void PrintCounter::endCalibCycle(Event& evt, Env& env)
{
}

void PrintCounter::endRun(Event& evt, Env& env)
{
}

void PrintCounter::endJob(Event& evt, Env& env)
{
}


PSANA_MODULE_FACTORY(PrintCounter);

}; // namespace psana_test
