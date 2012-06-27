#ifndef PSANA_ENVGETTER_H
#define PSANA_ENVGETTER_H

#include <boost/python/class.hpp>
#include <psddl_python/GetterMap.h>
#include <PSEnv/EnvObjectStore.h>

namespace Psana {
  using boost::python::api::object;
  using std::string;
  using PSEnv::EnvObjectStore;
  using PSEvt::Source;
  using Pds::Src;

  class EnvGetter : public Getter { // XXX should probably be EnvObjectStoreGetter
  public:
    static void addGetter(EnvGetter* getter) { envGetterMap.addGetter(getter); }

    virtual object get(EnvObjectStore& store, const Source& source, Src* foundSrc) = 0;

    static object get(string typeName, EnvObjectStore& store, const Source& source, Src* foundSrc) {
      EnvGetter* getter = (EnvGetter*) envGetterMap.getGetter(typeName);
      if (getter) {
        return getter->get(store, source, foundSrc);
      }
      int versionMin, versionMax;
      const char* _template = envGetterMap.getTemplate(typeName, &versionMin, &versionMax);
      if (! _template) {
        return object();
      }
      for (int version = versionMin; version <= versionMax; version++) {
        char vTypeName[128];
        sprintf(vTypeName, _template, version);
        getter = (EnvGetter*) envGetterMap.getGetter(vTypeName);
        if (getter) {
          object result(getter->get(store, source, foundSrc));
          if (result != object()) {
            return result;
          }
          printf("EnvGetter: Tried %s but it returned null\n", vTypeName);
        } else {
          printf("EnvGetter: %s does not exist\n", vTypeName);
        }
      }
      printf("EnvGetter: found no object for %s\n", _template);
      return object();
    }
  };
}
#endif // PSANA_ENVGETTER_H
