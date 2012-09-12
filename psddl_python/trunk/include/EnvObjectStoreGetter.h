#ifndef PSANA_ENVOBJECTSTOREGETTER_H
#define PSANA_ENVOBJECTSTOREGETTER_H

#include <boost/python/class.hpp>
#include <psddl_python/GetterMap.h>
#include <PSEnv/EnvObjectStore.h>

namespace psddl_python {
  using boost::python::api::object;
  using PSEnv::EnvObjectStore;
  using PSEvt::Source;

  class EnvObjectStoreGetter : public Getter {
  public:
    static void addGetter(EnvObjectStoreGetter* getter) { envObjectStoreGetterMap.addGetter(getter); }
    static object get(string typeName, EnvObjectStore& store, const Source& source, Src* foundSrc);
    virtual object get(EnvObjectStore& store, const Source& source, Src* foundSrc) = 0;
  };
}
#endif // PSANA_ENVOBJECTSTOREGETTER_H
