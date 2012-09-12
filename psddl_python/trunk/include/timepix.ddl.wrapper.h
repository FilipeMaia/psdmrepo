/* Do not edit this file, as it is auto-generated */

#ifndef PSDDL_PYTHON_TIMEPIX_DDL_WRAPPER_H
#define PSDDL_PYTHON_TIMEPIX_DDL_WRAPPER_H 1

#include <vector>
#include <ndarray/ndarray.h>
#include <pdsdata/xtc/TypeId.hh>
#include <psddl_python/DdlWrapper.h>

namespace psddl_python {
namespace Timepix {

using namespace boost::python;
using boost::python::api::object;
using boost::shared_ptr;
using std::vector;

extern void createWrappers();

class ConfigV1_Wrapper {
  shared_ptr<Psana::Timepix::ConfigV1> _o;
  Psana::Timepix::ConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_TimepixConfig };
  enum { Version = 1 };
  ConfigV1_Wrapper(shared_ptr<Psana::Timepix::ConfigV1> obj) : _o(obj), o(_o.get()) {}
  ConfigV1_Wrapper(Psana::Timepix::ConfigV1* obj) : o(obj) {}
  Psana::Timepix::ConfigV1::ReadoutSpeed readoutSpeed() const { return o->readoutSpeed(); }
  Psana::Timepix::ConfigV1::TriggerMode triggerMode() const { return o->triggerMode(); }
  int32_t shutterTimeout() const { return o->shutterTimeout(); }
  int32_t dac0Ikrum() const { return o->dac0Ikrum(); }
  int32_t dac0Disc() const { return o->dac0Disc(); }
  int32_t dac0Preamp() const { return o->dac0Preamp(); }
  int32_t dac0BufAnalogA() const { return o->dac0BufAnalogA(); }
  int32_t dac0BufAnalogB() const { return o->dac0BufAnalogB(); }
  int32_t dac0Hist() const { return o->dac0Hist(); }
  int32_t dac0ThlFine() const { return o->dac0ThlFine(); }
  int32_t dac0ThlCourse() const { return o->dac0ThlCourse(); }
  int32_t dac0Vcas() const { return o->dac0Vcas(); }
  int32_t dac0Fbk() const { return o->dac0Fbk(); }
  int32_t dac0Gnd() const { return o->dac0Gnd(); }
  int32_t dac0Ths() const { return o->dac0Ths(); }
  int32_t dac0BiasLvds() const { return o->dac0BiasLvds(); }
  int32_t dac0RefLvds() const { return o->dac0RefLvds(); }
  int32_t dac1Ikrum() const { return o->dac1Ikrum(); }
  int32_t dac1Disc() const { return o->dac1Disc(); }
  int32_t dac1Preamp() const { return o->dac1Preamp(); }
  int32_t dac1BufAnalogA() const { return o->dac1BufAnalogA(); }
  int32_t dac1BufAnalogB() const { return o->dac1BufAnalogB(); }
  int32_t dac1Hist() const { return o->dac1Hist(); }
  int32_t dac1ThlFine() const { return o->dac1ThlFine(); }
  int32_t dac1ThlCourse() const { return o->dac1ThlCourse(); }
  int32_t dac1Vcas() const { return o->dac1Vcas(); }
  int32_t dac1Fbk() const { return o->dac1Fbk(); }
  int32_t dac1Gnd() const { return o->dac1Gnd(); }
  int32_t dac1Ths() const { return o->dac1Ths(); }
  int32_t dac1BiasLvds() const { return o->dac1BiasLvds(); }
  int32_t dac1RefLvds() const { return o->dac1RefLvds(); }
  int32_t dac2Ikrum() const { return o->dac2Ikrum(); }
  int32_t dac2Disc() const { return o->dac2Disc(); }
  int32_t dac2Preamp() const { return o->dac2Preamp(); }
  int32_t dac2BufAnalogA() const { return o->dac2BufAnalogA(); }
  int32_t dac2BufAnalogB() const { return o->dac2BufAnalogB(); }
  int32_t dac2Hist() const { return o->dac2Hist(); }
  int32_t dac2ThlFine() const { return o->dac2ThlFine(); }
  int32_t dac2ThlCourse() const { return o->dac2ThlCourse(); }
  int32_t dac2Vcas() const { return o->dac2Vcas(); }
  int32_t dac2Fbk() const { return o->dac2Fbk(); }
  int32_t dac2Gnd() const { return o->dac2Gnd(); }
  int32_t dac2Ths() const { return o->dac2Ths(); }
  int32_t dac2BiasLvds() const { return o->dac2BiasLvds(); }
  int32_t dac2RefLvds() const { return o->dac2RefLvds(); }
  int32_t dac3Ikrum() const { return o->dac3Ikrum(); }
  int32_t dac3Disc() const { return o->dac3Disc(); }
  int32_t dac3Preamp() const { return o->dac3Preamp(); }
  int32_t dac3BufAnalogA() const { return o->dac3BufAnalogA(); }
  int32_t dac3BufAnalogB() const { return o->dac3BufAnalogB(); }
  int32_t dac3Hist() const { return o->dac3Hist(); }
  int32_t dac3ThlFine() const { return o->dac3ThlFine(); }
  int32_t dac3ThlCourse() const { return o->dac3ThlCourse(); }
  int32_t dac3Vcas() const { return o->dac3Vcas(); }
  int32_t dac3Fbk() const { return o->dac3Fbk(); }
  int32_t dac3Gnd() const { return o->dac3Gnd(); }
  int32_t dac3Ths() const { return o->dac3Ths(); }
  int32_t dac3BiasLvds() const { return o->dac3BiasLvds(); }
  int32_t dac3RefLvds() const { return o->dac3RefLvds(); }
};

class ConfigV2_Wrapper {
  shared_ptr<Psana::Timepix::ConfigV2> _o;
  Psana::Timepix::ConfigV2* o;
public:
  enum { TypeId = Pds::TypeId::Id_TimepixConfig };
  enum { Version = 2 };
  ConfigV2_Wrapper(shared_ptr<Psana::Timepix::ConfigV2> obj) : _o(obj), o(_o.get()) {}
  ConfigV2_Wrapper(Psana::Timepix::ConfigV2* obj) : o(obj) {}
  Psana::Timepix::ConfigV2::ReadoutSpeed readoutSpeed() const { return o->readoutSpeed(); }
  Psana::Timepix::ConfigV2::TriggerMode triggerMode() const { return o->triggerMode(); }
  int32_t timepixSpeed() const { return o->timepixSpeed(); }
  int32_t dac0Ikrum() const { return o->dac0Ikrum(); }
  int32_t dac0Disc() const { return o->dac0Disc(); }
  int32_t dac0Preamp() const { return o->dac0Preamp(); }
  int32_t dac0BufAnalogA() const { return o->dac0BufAnalogA(); }
  int32_t dac0BufAnalogB() const { return o->dac0BufAnalogB(); }
  int32_t dac0Hist() const { return o->dac0Hist(); }
  int32_t dac0ThlFine() const { return o->dac0ThlFine(); }
  int32_t dac0ThlCourse() const { return o->dac0ThlCourse(); }
  int32_t dac0Vcas() const { return o->dac0Vcas(); }
  int32_t dac0Fbk() const { return o->dac0Fbk(); }
  int32_t dac0Gnd() const { return o->dac0Gnd(); }
  int32_t dac0Ths() const { return o->dac0Ths(); }
  int32_t dac0BiasLvds() const { return o->dac0BiasLvds(); }
  int32_t dac0RefLvds() const { return o->dac0RefLvds(); }
  int32_t dac1Ikrum() const { return o->dac1Ikrum(); }
  int32_t dac1Disc() const { return o->dac1Disc(); }
  int32_t dac1Preamp() const { return o->dac1Preamp(); }
  int32_t dac1BufAnalogA() const { return o->dac1BufAnalogA(); }
  int32_t dac1BufAnalogB() const { return o->dac1BufAnalogB(); }
  int32_t dac1Hist() const { return o->dac1Hist(); }
  int32_t dac1ThlFine() const { return o->dac1ThlFine(); }
  int32_t dac1ThlCourse() const { return o->dac1ThlCourse(); }
  int32_t dac1Vcas() const { return o->dac1Vcas(); }
  int32_t dac1Fbk() const { return o->dac1Fbk(); }
  int32_t dac1Gnd() const { return o->dac1Gnd(); }
  int32_t dac1Ths() const { return o->dac1Ths(); }
  int32_t dac1BiasLvds() const { return o->dac1BiasLvds(); }
  int32_t dac1RefLvds() const { return o->dac1RefLvds(); }
  int32_t dac2Ikrum() const { return o->dac2Ikrum(); }
  int32_t dac2Disc() const { return o->dac2Disc(); }
  int32_t dac2Preamp() const { return o->dac2Preamp(); }
  int32_t dac2BufAnalogA() const { return o->dac2BufAnalogA(); }
  int32_t dac2BufAnalogB() const { return o->dac2BufAnalogB(); }
  int32_t dac2Hist() const { return o->dac2Hist(); }
  int32_t dac2ThlFine() const { return o->dac2ThlFine(); }
  int32_t dac2ThlCourse() const { return o->dac2ThlCourse(); }
  int32_t dac2Vcas() const { return o->dac2Vcas(); }
  int32_t dac2Fbk() const { return o->dac2Fbk(); }
  int32_t dac2Gnd() const { return o->dac2Gnd(); }
  int32_t dac2Ths() const { return o->dac2Ths(); }
  int32_t dac2BiasLvds() const { return o->dac2BiasLvds(); }
  int32_t dac2RefLvds() const { return o->dac2RefLvds(); }
  int32_t dac3Ikrum() const { return o->dac3Ikrum(); }
  int32_t dac3Disc() const { return o->dac3Disc(); }
  int32_t dac3Preamp() const { return o->dac3Preamp(); }
  int32_t dac3BufAnalogA() const { return o->dac3BufAnalogA(); }
  int32_t dac3BufAnalogB() const { return o->dac3BufAnalogB(); }
  int32_t dac3Hist() const { return o->dac3Hist(); }
  int32_t dac3ThlFine() const { return o->dac3ThlFine(); }
  int32_t dac3ThlCourse() const { return o->dac3ThlCourse(); }
  int32_t dac3Vcas() const { return o->dac3Vcas(); }
  int32_t dac3Fbk() const { return o->dac3Fbk(); }
  int32_t dac3Gnd() const { return o->dac3Gnd(); }
  int32_t dac3Ths() const { return o->dac3Ths(); }
  int32_t dac3BiasLvds() const { return o->dac3BiasLvds(); }
  int32_t dac3RefLvds() const { return o->dac3RefLvds(); }
  int32_t driverVersion() const { return o->driverVersion(); }
  uint32_t firmwareVersion() const { return o->firmwareVersion(); }
  uint32_t pixelThreshSize() const { return o->pixelThreshSize(); }
  PyObject* pixelThresh() const { ND_CONVERT(o->pixelThresh(), uint8_t, 1); }
  const char* chip0Name() const { return o->chip0Name(); }
  const char* chip1Name() const { return o->chip1Name(); }
  const char* chip2Name() const { return o->chip2Name(); }
  const char* chip3Name() const { return o->chip3Name(); }
  int32_t chip0ID() const { return o->chip0ID(); }
  int32_t chip1ID() const { return o->chip1ID(); }
  int32_t chip2ID() const { return o->chip2ID(); }
  int32_t chip3ID() const { return o->chip3ID(); }
  int32_t chipCount() const { return o->chipCount(); }
};

class DataV1_Wrapper {
  shared_ptr<Psana::Timepix::DataV1> _o;
  Psana::Timepix::DataV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_TimepixData };
  enum { Version = 1 };
  DataV1_Wrapper(shared_ptr<Psana::Timepix::DataV1> obj) : _o(obj), o(_o.get()) {}
  DataV1_Wrapper(Psana::Timepix::DataV1* obj) : o(obj) {}
  uint32_t timestamp() const { return o->timestamp(); }
  uint16_t frameCounter() const { return o->frameCounter(); }
  uint16_t lostRows() const { return o->lostRows(); }
  PyObject* data() const { ND_CONVERT(o->data(), uint16_t, 2); }
  uint32_t width() const { return o->width(); }
  uint32_t height() const { return o->height(); }
  uint32_t depth() const { return o->depth(); }
  uint32_t depth_bytes() const { return o->depth_bytes(); }
};

class DataV2_Wrapper {
  shared_ptr<Psana::Timepix::DataV2> _o;
  Psana::Timepix::DataV2* o;
public:
  enum { TypeId = Pds::TypeId::Id_TimepixData };
  enum { Version = 2 };
  DataV2_Wrapper(shared_ptr<Psana::Timepix::DataV2> obj) : _o(obj), o(_o.get()) {}
  DataV2_Wrapper(Psana::Timepix::DataV2* obj) : o(obj) {}
  uint16_t width() const { return o->width(); }
  uint16_t height() const { return o->height(); }
  uint32_t timestamp() const { return o->timestamp(); }
  uint16_t frameCounter() const { return o->frameCounter(); }
  uint16_t lostRows() const { return o->lostRows(); }
  PyObject* data() const { ND_CONVERT(o->data(), uint16_t, 2); }
  uint32_t depth() const { return o->depth(); }
  uint32_t depth_bytes() const { return o->depth_bytes(); }
};

  class ConfigV1_Getter : public psddl_python::EnvObjectStoreGetter {
  public:
  const char* getTypeName() { return "Psana::Timepix::ConfigV1";}
  const char* getGetterClassName() { return "psddl_python::EnvObjectStoreGetter";}
    int getVersion() {
      return Psana::Timepix::ConfigV1::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& source, Pds::Src* foundSrc) {
      boost::shared_ptr<Psana::Timepix::ConfigV1> result = store.get(source, foundSrc);
      return result.get() ? object(ConfigV1_Wrapper(result)) : object();
    }
  };

  class ConfigV2_Getter : public psddl_python::EnvObjectStoreGetter {
  public:
  const char* getTypeName() { return "Psana::Timepix::ConfigV2";}
  const char* getGetterClassName() { return "psddl_python::EnvObjectStoreGetter";}
    int getVersion() {
      return Psana::Timepix::ConfigV2::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& source, Pds::Src* foundSrc) {
      boost::shared_ptr<Psana::Timepix::ConfigV2> result = store.get(source, foundSrc);
      return result.get() ? object(ConfigV2_Wrapper(result)) : object();
    }
  };

  class DataV1_Getter : public psddl_python::EventGetter {
  public:
  const char* getTypeName() { return "Psana::Timepix::DataV1";}
  const char* getGetterClassName() { return "psddl_python::EventGetter";}
    int getVersion() {
      return Psana::Timepix::DataV1::Version;
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<Psana::Timepix::DataV1> result = evt.get(source, key, foundSrc);
      return result.get() ? object(DataV1_Wrapper(result)) : object();
    }
  };

  class DataV2_Getter : public psddl_python::EventGetter {
  public:
  const char* getTypeName() { return "Psana::Timepix::DataV2";}
  const char* getGetterClassName() { return "psddl_python::EventGetter";}
    int getVersion() {
      return Psana::Timepix::DataV2::Version;
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<Psana::Timepix::DataV2> result = evt.get(source, key, foundSrc);
      return result.get() ? object(DataV2_Wrapper(result)) : object();
    }
  };
} // namespace Timepix
} // namespace psddl_python
#endif // PSDDL_PYTHON_TIMEPIX_DDL_WRAPPER_H
