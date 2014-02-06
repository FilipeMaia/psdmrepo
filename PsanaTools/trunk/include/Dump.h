#include "psana/Module.h"

namespace PsanaTools {

class Dump : public Module {
 public:
  Dump(std::string);
  virtual ~Dump();

  virtual void beginJob(Event& evt, Env& env);
  virtual void beginRun(Event& evt, Env& env);
  virtual void beginCalibCycle(Event& evt, Env& env);
  virtual void event(Event& evt, Env& env);
  virtual void endCalibCycle(Event& evt, Env& env);
  virtual void endRun(Event& evt, Env& env);
  virtual void endJob(Event& evt, Env& env);
}; // class Dump

}; // namespace PsanaTools
