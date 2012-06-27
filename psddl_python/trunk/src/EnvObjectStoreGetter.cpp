#include <psddl_python/EnvObjectStoreGetter.h>

namespace Psana {
  object EnvObjectStoreGetter::get(string typeName, EnvObjectStore& store, const Source& source, Src* foundSrc) {
    EnvObjectStoreGetter* getter = (EnvObjectStoreGetter*) envObjectStoreGetterMap.getGetter(typeName);
    if (getter) {
      return getter->get(store, source, foundSrc);
    }
    int versionMin, versionMax;
    const char* _template = envObjectStoreGetterMap.getTemplate(typeName, &versionMin, &versionMax);
    if (! _template) {
      return object();
    }
    for (int version = versionMin; version <= versionMax; version++) {
      char vTypeName[128];
      sprintf(vTypeName, _template, version);
      getter = (EnvObjectStoreGetter*) envObjectStoreGetterMap.getGetter(vTypeName);
      if (getter) {
        object result(getter->get(store, source, foundSrc));
        if (result != object()) {
          return result;
        }
        printf("EnvObjectStoreGetter: Tried %s but it returned null\n", vTypeName);
      } else {
        printf("EnvObjectStoreGetter: %s does not exist\n", vTypeName);
      }
    }
    printf("EnvObjectStoreGetter: found no object for %s\n", _template);
    return object();
  }
}
