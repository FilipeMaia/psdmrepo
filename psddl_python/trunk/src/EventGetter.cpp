#include <psddl_python/EventGetter.h>

namespace Psana {
  object EventGetter::get(string typeName, Event& event, Source& source, const std::string& key, Src* foundSrc) {
    EventGetter* getter = (EventGetter*) eventGetterMap.getGetter(typeName);
    if (getter) {
      return getter->get(event, source, key, foundSrc);
    }
    int versionMin, versionMax;
    const char* _template = eventGetterMap.getTemplate(typeName, &versionMin, &versionMax);
    if (! _template) {
      printf("EventGetter: %s does not exist\n", typeName.c_str());
      return object();
    }
    for (int version = versionMin; version <= versionMax; version++) {
      char vTypeName[128];
      sprintf(vTypeName, _template, version);
      getter = (EventGetter*) eventGetterMap.getGetter(vTypeName);
      if (getter) {
        object result(getter->get(event, source, key, foundSrc));
        if (result != object()) {
          printf("EventGetter: returning result of type %s\n", vTypeName);
          return result;
        }
        //printf("EventGetter: tried %s but it returned null\n", vTypeName);
      } else {
        printf("EventGetter: %s does not exist\n", vTypeName);
      }
    }
    printf("EventGetter: found no object for %s\n", _template);
    return object();
  }
}
