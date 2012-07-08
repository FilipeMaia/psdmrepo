/* Do not edit this file, as it is auto-generated */

#ifndef PSANA_IPIMB_DDL_WRAPPER_H
#define PSANA_IPIMB_DDL_WRAPPER_H 1

#include <vector>
#include <ndarray/ndarray.h>
#include <pdsdata/xtc/TypeId.hh>
#include <psddl_python/DdlWrapper.h>

namespace Psana {
namespace Ipimb {

using namespace boost::python;
using boost::python::api::object;
using boost::shared_ptr;
using std::vector;

extern void createWrappers();

class ConfigV1_Wrapper {
  shared_ptr<ConfigV1> _o;
  ConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_IpimbConfig };
  enum { Version = 1 };
  ConfigV1_Wrapper(shared_ptr<ConfigV1> obj) : _o(obj), o(_o.get()) {}
  ConfigV1_Wrapper(ConfigV1* obj) : o(obj) {}
  uint64_t triggerCounter() const { return o->triggerCounter(); }
  uint64_t serialID() const { return o->serialID(); }
  uint16_t chargeAmpRange() const { return o->chargeAmpRange(); }
  uint16_t calibrationRange() const { return o->calibrationRange(); }
  uint32_t resetLength() const { return o->resetLength(); }
  uint32_t resetDelay() const { return o->resetDelay(); }
  float chargeAmpRefVoltage() const { return o->chargeAmpRefVoltage(); }
  float calibrationVoltage() const { return o->calibrationVoltage(); }
  float diodeBias() const { return o->diodeBias(); }
  uint16_t status() const { return o->status(); }
  uint16_t errors() const { return o->errors(); }
  uint16_t calStrobeLength() const { return o->calStrobeLength(); }
  uint32_t trigDelay() const { return o->trigDelay(); }
  Ipimb::ConfigV1::CapacitorValue diodeGain(uint32_t ch) const { return o->diodeGain(ch); }
};

class ConfigV2_Wrapper {
  shared_ptr<ConfigV2> _o;
  ConfigV2* o;
public:
  enum { TypeId = Pds::TypeId::Id_IpimbConfig };
  enum { Version = 2 };
  ConfigV2_Wrapper(shared_ptr<ConfigV2> obj) : _o(obj), o(_o.get()) {}
  ConfigV2_Wrapper(ConfigV2* obj) : o(obj) {}
  uint64_t triggerCounter() const { return o->triggerCounter(); }
  uint64_t serialID() const { return o->serialID(); }
  uint16_t chargeAmpRange() const { return o->chargeAmpRange(); }
  uint16_t calibrationRange() const { return o->calibrationRange(); }
  uint32_t resetLength() const { return o->resetLength(); }
  uint32_t resetDelay() const { return o->resetDelay(); }
  float chargeAmpRefVoltage() const { return o->chargeAmpRefVoltage(); }
  float calibrationVoltage() const { return o->calibrationVoltage(); }
  float diodeBias() const { return o->diodeBias(); }
  uint16_t status() const { return o->status(); }
  uint16_t errors() const { return o->errors(); }
  uint16_t calStrobeLength() const { return o->calStrobeLength(); }
  uint32_t trigDelay() const { return o->trigDelay(); }
  uint32_t trigPsDelay() const { return o->trigPsDelay(); }
  uint32_t adcDelay() const { return o->adcDelay(); }
  Ipimb::ConfigV2::CapacitorValue diodeGain(uint32_t ch) const { return o->diodeGain(ch); }
};

class DataV1_Wrapper {
  shared_ptr<DataV1> _o;
  DataV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_IpimbData };
  enum { Version = 1 };
  DataV1_Wrapper(shared_ptr<DataV1> obj) : _o(obj), o(_o.get()) {}
  DataV1_Wrapper(DataV1* obj) : o(obj) {}
  uint64_t triggerCounter() const { return o->triggerCounter(); }
  uint16_t config0() const { return o->config0(); }
  uint16_t config1() const { return o->config1(); }
  uint16_t config2() const { return o->config2(); }
  uint16_t channel0() const { return o->channel0(); }
  uint16_t channel1() const { return o->channel1(); }
  uint16_t channel2() const { return o->channel2(); }
  uint16_t channel3() const { return o->channel3(); }
  uint16_t checksum() const { return o->checksum(); }
  float channel0Volts() const { return o->channel0Volts(); }
  float channel1Volts() const { return o->channel1Volts(); }
  float channel2Volts() const { return o->channel2Volts(); }
  float channel3Volts() const { return o->channel3Volts(); }
};

class DataV2_Wrapper {
  shared_ptr<DataV2> _o;
  DataV2* o;
public:
  enum { TypeId = Pds::TypeId::Id_IpimbData };
  enum { Version = 2 };
  DataV2_Wrapper(shared_ptr<DataV2> obj) : _o(obj), o(_o.get()) {}
  DataV2_Wrapper(DataV2* obj) : o(obj) {}
  uint16_t config0() const { return o->config0(); }
  uint16_t config1() const { return o->config1(); }
  uint16_t config2() const { return o->config2(); }
  uint16_t channel0() const { return o->channel0(); }
  uint16_t channel1() const { return o->channel1(); }
  uint16_t channel2() const { return o->channel2(); }
  uint16_t channel3() const { return o->channel3(); }
  uint16_t channel0ps() const { return o->channel0ps(); }
  uint16_t channel1ps() const { return o->channel1ps(); }
  uint16_t channel2ps() const { return o->channel2ps(); }
  uint16_t channel3ps() const { return o->channel3ps(); }
  uint16_t checksum() const { return o->checksum(); }
  float channel0Volts() const { return o->channel0Volts(); }
  float channel1Volts() const { return o->channel1Volts(); }
  float channel2Volts() const { return o->channel2Volts(); }
  float channel3Volts() const { return o->channel3Volts(); }
  float channel0psVolts() const { return o->channel0psVolts(); }
  float channel1psVolts() const { return o->channel1psVolts(); }
  float channel2psVolts() const { return o->channel2psVolts(); }
  float channel3psVolts() const { return o->channel3psVolts(); }
  uint64_t triggerCounter() const { return o->triggerCounter(); }
};

  class ConfigV1_Getter : public Psana::EnvObjectStoreGetter {
  public:
  const char* getTypeName() { return "Psana::Ipimb::ConfigV1";}
  const char* getGetterClassName() { return "Psana::EnvObjectStoreGetter";}
    int getVersion() {
      return ConfigV1::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& source, Pds::Src* foundSrc) {
      boost::shared_ptr<ConfigV1> result = store.get(source, foundSrc);
      return result.get() ? object(ConfigV1_Wrapper(result)) : object();
    }
  };

  class ConfigV2_Getter : public Psana::EnvObjectStoreGetter {
  public:
  const char* getTypeName() { return "Psana::Ipimb::ConfigV2";}
  const char* getGetterClassName() { return "Psana::EnvObjectStoreGetter";}
    int getVersion() {
      return ConfigV2::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& source, Pds::Src* foundSrc) {
      boost::shared_ptr<ConfigV2> result = store.get(source, foundSrc);
      return result.get() ? object(ConfigV2_Wrapper(result)) : object();
    }
  };

  class DataV1_Getter : public Psana::EventGetter {
  public:
  const char* getTypeName() { return "Psana::Ipimb::DataV1";}
  const char* getGetterClassName() { return "Psana::EventGetter";}
    int getVersion() {
      return DataV1::Version;
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<DataV1> result = evt.get(source, key, foundSrc);
      return result.get() ? object(DataV1_Wrapper(result)) : object();
    }
  };

  class DataV2_Getter : public Psana::EventGetter {
  public:
  const char* getTypeName() { return "Psana::Ipimb::DataV2";}
  const char* getGetterClassName() { return "Psana::EventGetter";}
    int getVersion() {
      return DataV2::Version;
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key, Pds::Src* foundSrc) {
      shared_ptr<DataV2> result = evt.get(source, key, foundSrc);
      return result.get() ? object(DataV2_Wrapper(result)) : object();
    }
  };
} // namespace Ipimb
} // namespace Psana
#endif // PSANA_IPIMB_DDL_WRAPPER_H
