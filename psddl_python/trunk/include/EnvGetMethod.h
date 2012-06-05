#ifndef PSANA_ENVGETMETHOD_H
#define PSANA_ENVGETMETHOD_H

#include <psddl_python/GenericGetter.h>
#include <PSEnv/Env.h>

namespace Psana {
  using PSEnv::EnvObjectStore;
  using PSEvt::Source;

  class EnvGetMethod : public GetMethod {
  private:
    EnvObjectStore& m_store;
    const Source& m_src;
  public:
    EnvGetMethod(EnvObjectStore& store, const Source& src) : m_store(store), m_src(src) {}
    object get(GenericGetter* getter) {
      return ((EnvGetter*) getter)->get(m_store, m_src);
    }
  };
}
#endif // PSANA_ENVGETMETHOD_H
