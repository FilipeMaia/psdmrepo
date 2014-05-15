#include "psana/Module.h"

namespace psana_test {

class PrintCounter : public Module {
 public:
  PrintCounter(std::string);
  virtual ~PrintCounter();

  virtual void beginJob(Event& evt, Env& env);
  virtual void beginRun(Event& evt, Env& env);
  virtual void beginCalibCycle(Event& evt, Env& env);
  virtual void event(Event& evt, Env& env);
  virtual void endCalibCycle(Event& evt, Env& env);
  virtual void endRun(Event& evt, Env& env);
  virtual void endJob(Event& evt, Env& env);
 private:
  long m_run, m_calibCycle, m_event;
}; // class PrintCounter

}; // namespace psana_test
