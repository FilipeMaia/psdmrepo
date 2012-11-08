/* Do not edit this file, as it is auto-generated */

#ifndef PSDDL_PYTHON_CONTROL_DDL_WRAPPER_H
#define PSDDL_PYTHON_CONTROL_DDL_WRAPPER_H 1

#include <psddl_python/DdlWrapper.h>
#include <vector>
#include <ndarray/ndarray.h>
#include <pdsdata/xtc/TypeId.hh>
#include <psddl_psana/control.ddl.h> // inc_psana

#include <pdsdata/xtc/ClockTime.hh>
namespace psddl_python {
namespace ControlData {

using namespace boost::python;
using boost::python::api::object;
using boost::shared_ptr;
using std::vector;

extern void createWrappers();

class PVControl_Wrapper {
  shared_ptr<Psana::ControlData::PVControl> _o;
  Psana::ControlData::PVControl* o;
public:
  PVControl_Wrapper(shared_ptr<Psana::ControlData::PVControl> obj) : _o(obj), o(_o.get()) {}
  PVControl_Wrapper(Psana::ControlData::PVControl* obj) : o(obj) {}
  const char* name() const { return o->name(); }
  uint32_t index() const { return o->index(); }
  double value() const { return o->value(); }
  uint8_t array() const { return o->array(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class PVMonitor_Wrapper {
  shared_ptr<Psana::ControlData::PVMonitor> _o;
  Psana::ControlData::PVMonitor* o;
public:
  PVMonitor_Wrapper(shared_ptr<Psana::ControlData::PVMonitor> obj) : _o(obj), o(_o.get()) {}
  PVMonitor_Wrapper(Psana::ControlData::PVMonitor* obj) : o(obj) {}
  const char* name() const { return o->name(); }
  uint32_t index() const { return o->index(); }
  double loValue() const { return o->loValue(); }
  double hiValue() const { return o->hiValue(); }
  uint8_t array() const { return o->array(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class PVLabel_Wrapper {
  shared_ptr<Psana::ControlData::PVLabel> _o;
  Psana::ControlData::PVLabel* o;
public:
  PVLabel_Wrapper(shared_ptr<Psana::ControlData::PVLabel> obj) : _o(obj), o(_o.get()) {}
  PVLabel_Wrapper(Psana::ControlData::PVLabel* obj) : o(obj) {}
  const char* name() const { return o->name(); }
  const char* value() const { return o->value(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class ConfigV1_Wrapper {
  shared_ptr<Psana::ControlData::ConfigV1> _o;
  Psana::ControlData::ConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_ControlConfig };
  enum { Version = 1 };
  ConfigV1_Wrapper(shared_ptr<Psana::ControlData::ConfigV1> obj) : _o(obj), o(_o.get()) {}
  ConfigV1_Wrapper(Psana::ControlData::ConfigV1* obj) : o(obj) {}
  uint32_t events() const { return o->events(); }
  uint8_t uses_duration() const { return o->uses_duration(); }
  uint8_t uses_events() const { return o->uses_events(); }
  const Pds::ClockTime& duration() const { return o->duration(); }
  uint32_t npvControls() const { return o->npvControls(); }
  uint32_t npvMonitors() const { return o->npvMonitors(); }
  vector<Psana::ControlData::PVControl> pvControls() const { VEC_CONVERT(o->pvControls(), Psana::ControlData::PVControl); }
  vector<Psana::ControlData::PVMonitor> pvMonitors() const { VEC_CONVERT(o->pvMonitors(), Psana::ControlData::PVMonitor); }
};

class ConfigV2_Wrapper {
  shared_ptr<Psana::ControlData::ConfigV2> _o;
  Psana::ControlData::ConfigV2* o;
public:
  enum { TypeId = Pds::TypeId::Id_ControlConfig };
  enum { Version = 2 };
  ConfigV2_Wrapper(shared_ptr<Psana::ControlData::ConfigV2> obj) : _o(obj), o(_o.get()) {}
  ConfigV2_Wrapper(Psana::ControlData::ConfigV2* obj) : o(obj) {}
  uint32_t events() const { return o->events(); }
  uint8_t uses_duration() const { return o->uses_duration(); }
  uint8_t uses_events() const { return o->uses_events(); }
  const Pds::ClockTime& duration() const { return o->duration(); }
  uint32_t npvControls() const { return o->npvControls(); }
  uint32_t npvMonitors() const { return o->npvMonitors(); }
  uint32_t npvLabels() const { return o->npvLabels(); }
  vector<Psana::ControlData::PVControl> pvControls() const { VEC_CONVERT(o->pvControls(), Psana::ControlData::PVControl); }
  vector<Psana::ControlData::PVMonitor> pvMonitors() const { VEC_CONVERT(o->pvMonitors(), Psana::ControlData::PVMonitor); }
  vector<Psana::ControlData::PVLabel> pvLabels() const { VEC_CONVERT(o->pvLabels(), Psana::ControlData::PVLabel); }
};

  class PVControl_Getter : public psddl_python::EventGetter {
  public:
  const char* getTypeName() { return "Psana::ControlData::PVControl";}
  const char* getGetterClassName() { return "psddl_python::EventGetter";}
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<Psana::ControlData::PVControl> result = evt.get(source, key, foundSrc);
      return result.get() ? object(PVControl_Wrapper(result)) : object();
    }
  };

  class PVMonitor_Getter : public psddl_python::EventGetter {
  public:
  const char* getTypeName() { return "Psana::ControlData::PVMonitor";}
  const char* getGetterClassName() { return "psddl_python::EventGetter";}
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<Psana::ControlData::PVMonitor> result = evt.get(source, key, foundSrc);
      return result.get() ? object(PVMonitor_Wrapper(result)) : object();
    }
  };

  class PVLabel_Getter : public psddl_python::EventGetter {
  public:
  const char* getTypeName() { return "Psana::ControlData::PVLabel";}
  const char* getGetterClassName() { return "psddl_python::EventGetter";}
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<Psana::ControlData::PVLabel> result = evt.get(source, key, foundSrc);
      return result.get() ? object(PVLabel_Wrapper(result)) : object();
    }
  };

  class ConfigV1_Getter : public psddl_python::EnvObjectStoreGetter {
  public:
  const char* getTypeName() { return "Psana::ControlData::ConfigV1";}
  const char* getGetterClassName() { return "psddl_python::EnvObjectStoreGetter";}
    int getVersion() {
      return Psana::ControlData::ConfigV1::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& source, Pds::Src* foundSrc) {
      boost::shared_ptr<Psana::ControlData::ConfigV1> result = store.get(source, foundSrc);
      return result.get() ? object(ConfigV1_Wrapper(result)) : object();
    }
  };

  class ConfigV2_Getter : public psddl_python::EnvObjectStoreGetter {
  public:
  const char* getTypeName() { return "Psana::ControlData::ConfigV2";}
  const char* getGetterClassName() { return "psddl_python::EnvObjectStoreGetter";}
    int getVersion() {
      return Psana::ControlData::ConfigV2::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& source, Pds::Src* foundSrc) {
      boost::shared_ptr<Psana::ControlData::ConfigV2> result = store.get(source, foundSrc);
      return result.get() ? object(ConfigV2_Wrapper(result)) : object();
    }
  };
} // namespace ControlData
} // namespace psddl_python
#endif // PSDDL_PYTHON_CONTROL_DDL_WRAPPER_H
