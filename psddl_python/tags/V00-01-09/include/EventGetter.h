#ifndef PSANA_EVENTGETTER_H
#define PSANA_EVENTGETTER_H

#include <boost/python/class.hpp>
#include <psddl_python/GetterMap.h>
#include <PSEvt/Event.h>

namespace psddl_python {
  using boost::python::api::object;
  using PSEvt::Event;
  using PSEvt::Source;
  using Pds::Src;

  class EventGetter : public Getter {
  public:
    static void addGetter(EventGetter* getter) { eventGetterMap.addGetter(getter); }
    static object get(string typeName, Event& event, Source& source, const std::string& key, Src* foundSrc);
    virtual object get(Event& event, Source& source, const std::string& key, Src* foundSrc) = 0;
  };
}

#endif // PSANA_EVENTGETTER_H
