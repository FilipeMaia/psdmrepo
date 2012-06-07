#ifndef PSANA_EVTGETMETHOD_H
#define PSANA_EVTGETMETHOD_H

#include <psddl_python/GenericGetter.h>
#include <psddl_python/EvtGetter.h>
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
    Src* m_foundSrc;
  public:
    EvtGetMethod(Event& event) : m_event(event), m_key(""), m_source(0), m_src(0), m_foundSrc(0) {}
    void addKey(string& key) { m_key = key; }
    void addSource(Source* source) { m_source = source; }
    void addSrc(Src* src) { m_src = src; }
    void addFoundSrc(Src* foundSrc) { m_foundSrc = foundSrc; }

    object get(GenericGetter* getter) {
      string key2(m_key);
      Src foundSrc;
      Src *foundSrcPtr = (m_foundSrc ? m_foundSrc : &foundSrc);
      if (m_src) {
        return ((EvtGetter*) getter)->get(m_event, *m_src, key2, foundSrcPtr);
      }
      if (m_source) {
        return ((EvtGetter*) getter)->get(m_event, *m_source, key2, foundSrcPtr);
      }
      return ((EvtGetter*) getter)->get(m_event, key2, foundSrcPtr);
    }
  };
}
#endif // PSANA_EVTGETMETHOD_H
