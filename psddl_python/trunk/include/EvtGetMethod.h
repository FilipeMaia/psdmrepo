#ifndef PSANA_EVTGETMETHOD_H
#define PSANA_EVTGETMETHOD_H

#include <psddl_python/GenericGetter.h>
#include <PSEvt/Event.h>

namespace Psana {
  using PSEvt::Event;
  using PSEvt::Source;
  using Pds::Src;

  class EvtGetMethod : public GetMethod {
  private:
    Event& m_event;
    string m_key;
    Source* m_source;
    Src* m_src;
  public:
    EvtGetMethod(Event& event, string key = string()) :
      m_event(event), m_key(key), m_source(0), m_src(0) {}
    EvtGetMethod(Event& event, Src& src, string key = string()) :
      m_event(event), m_key(key), m_source(0), m_src(&src) {}
    EvtGetMethod(Event& event, Source& source, string key = string()) :
      m_event(event), m_key(key), m_source(&source), m_src() {}

    object get(GenericGetter* getter) {
      string key2(m_key);
      if (m_src) {
        return ((EvtGetter*) getter)->get(m_event, *m_src, key2);
      }
      if (m_source) {
        return ((EvtGetter*) getter)->get(m_event, *m_source, key2);
      }
      return ((EvtGetter*) getter)->get(m_event, key2);
    }
  };
}
#endif // PSANA_EVTGETMETHOD_H
