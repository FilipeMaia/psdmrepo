#ifndef PSANA_ENVGETTER_H
#define PSANA_ENVGETTER_H

#include <psddl_python/GenericGetter.h>
#include <PSEnv/Env.h>

namespace Psana {
  using PSEnv::EnvObjectStore;
  using PSEvt::Source;

  class EnvGetter : public GenericGetter {
  public:
    virtual object get(EnvObjectStore& store, const Source& src) = 0;
  };
}
#endif // PSANA_ENVGETTER_H
