#ifndef PSANA_EVENTGETTER_H
#define PSANA_EVENTGETTER_H

#include <boost/python/class.hpp>
#include <psddl_python/GetterMap.h>
#include <PSEvt/Event.h>

namespace Psana {
  using boost::python::api::object;
  using PSEvt::Event;
  using PSEvt::Source;
  using Pds::Src;

  class EventGetter : public Getter {
  public:
    static void addGetter(EventGetter* getter) { eventGetterMap.addGetter(getter); }

    virtual object get(Event& event, Source& source, const std::string& key, Src* foundSrc) = 0;

    static object get(string typeName, Event& event, Source& source, const std::string& key, Src* foundSrc) {
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
            return result;
          }
          printf("EventGetter: tried %s but it returned null\n", vTypeName);
        } else {
          printf("EventGetter: %s does not exist\n", vTypeName);
        }
      }
      printf("EventGetter: found no object for %s\n", _template);
      return object();
    }
  };
}

#endif // PSANA_EVENTGETTER_H
