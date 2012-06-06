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
    Pds::Src* m_foundSrc;
  public:
    EnvGetMethod(EnvObjectStore& store, const Source& src, Pds::Src* foundSrc) : m_store(store), m_src(src), m_foundSrc(foundSrc) {}
    object get(GenericGetter* getter) {
      return ((EnvGetter*) getter)->get(m_store, m_src, m_foundSrc);
    }
  };
}
#endif // PSANA_ENVGETMETHOD_H
