#include "psana/Module.h"

namespace psana_test {

class Dump : public Module {
 public:
  Dump(std::string);
  virtual ~Dump();

  virtual void beginJob(Event& evt, Env& env);
  virtual void event(Event& evt, Env& env);
}; // class Dump

}; // namespace psana_test
