/* Do not edit this file, as it is auto-generated */

#ifndef PSANA_CONTROL_DDL_WRAPPER_H
#define PSANA_CONTROL_DDL_WRAPPER_H 1

#include <vector>
#include <ndarray/ndarray.h>
#include <pdsdata/xtc/TypeId.hh>
#include <psddl_python/DdlWrapper.h>

#include <pdsdata/xtc/ClockTime.hh>
namespace Psana {
namespace ControlData {

using namespace boost::python;
using boost::python::api::object;
using boost::shared_ptr;
using std::vector;

extern void createWrappers();

class PVControl_Wrapper {
  shared_ptr<PVControl> _o;
  PVControl* o;
public:
  PVControl_Wrapper(shared_ptr<PVControl> obj) : _o(obj), o(_o.get()) {}
  PVControl_Wrapper(PVControl* obj) : o(obj) {}
  const char* name() const { return o->name(); }
  uint32_t index() const { return o->index(); }
  double value() const { return o->value(); }
  uint8_t array() const { return o->array(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class PVMonitor_Wrapper {
  shared_ptr<PVMonitor> _o;
  PVMonitor* o;
public:
  PVMonitor_Wrapper(shared_ptr<PVMonitor> obj) : _o(obj), o(_o.get()) {}
  PVMonitor_Wrapper(PVMonitor* obj) : o(obj) {}
  const char* name() const { return o->name(); }
  uint32_t index() const { return o->index(); }
  double loValue() const { return o->loValue(); }
  double hiValue() const { return o->hiValue(); }
  uint8_t array() const { return o->array(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class ConfigV1_Wrapper {
  shared_ptr<ConfigV1> _o;
  ConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_ControlConfig };
  enum { Version = 1 };
  ConfigV1_Wrapper(shared_ptr<ConfigV1> obj) : _o(obj), o(_o.get()) {}
  ConfigV1_Wrapper(ConfigV1* obj) : o(obj) {}
  uint32_t events() const { return o->events(); }
  uint8_t uses_duration() const { return o->uses_duration(); }
  uint8_t uses_events() const { return o->uses_events(); }
  const Pds::ClockTime& duration() const { return o->duration(); }
  uint32_t npvControls() const { return o->npvControls(); }
  uint32_t npvMonitors() const { return o->npvMonitors(); }
  vector<ControlData::PVControl> pvControls() const { VEC_CONVERT(o->pvControls(), ControlData::PVControl); }
  vector<ControlData::PVMonitor> pvMonitors() const { VEC_CONVERT(o->pvMonitors(), ControlData::PVMonitor); }
};

  class PVControl_Getter : public Psana::EventGetter {
  public:
  const char* getTypeName() { return "Psana::ControlData::PVControl";}
  const char* getGetterClassName() { return "Psana::EventGetter";}
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<PVControl> result = evt.get(source, key, foundSrc);
      return result.get() ? object(PVControl_Wrapper(result)) : object();
    }
  };

  class PVMonitor_Getter : public Psana::EventGetter {
  public:
  const char* getTypeName() { return "Psana::ControlData::PVMonitor";}
  const char* getGetterClassName() { return "Psana::EventGetter";}
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<PVMonitor> result = evt.get(source, key, foundSrc);
      return result.get() ? object(PVMonitor_Wrapper(result)) : object();
    }
  };

  class ConfigV1_Getter : public Psana::EnvObjectStoreGetter {
  public:
  const char* getTypeName() { return "Psana::ControlData::ConfigV1";}
  const char* getGetterClassName() { return "Psana::EnvObjectStoreGetter";}
    int getVersion() {
      return ConfigV1::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& source, Pds::Src* foundSrc) {
      boost::shared_ptr<ConfigV1> result = store.get(source, foundSrc);
      return result.get() ? object(ConfigV1_Wrapper(result)) : object();
    }
  };
} // namespace ControlData
} // namespace Psana
#endif // PSANA_CONTROL_DDL_WRAPPER_H
