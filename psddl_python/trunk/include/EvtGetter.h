#ifndef PSANA_EVTGETTER_H
#define PSANA_EVTGETTER_H

#include <string>
#include <boost/python/class.hpp>
#include <psddl_python/GenericGetter.h>
#include <PSEvt/Event.h>

namespace Psana {
  using boost::python::api::object;
  using std::string;
  using PSEvt::Event;
  using PSEvt::Source;
  using Pds::Src;

  class EvtGetter : public GenericGetter {
  public:
    virtual object get(Event& evt, const string& key=string(), Src* foundSrc=0) = 0;
    virtual object get(Event& evt, Src& src, const string& key=string(), Src* foundSrc=0) = 0;
    virtual object get(Event& evt, PSEvt::Source& source, const string& key=string(), Src* foundSrc=0) = 0;
  };
}

#endif // PSANA_EVTGETTER_H
