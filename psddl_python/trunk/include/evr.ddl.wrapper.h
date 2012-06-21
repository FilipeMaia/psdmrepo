/* Do not edit this file, as it is auto-generated */

#ifndef PSANA_EVR_DDL_WRAPPER_H
#define PSANA_EVR_DDL_WRAPPER_H 1

#include <vector>
#include <ndarray/ndarray.h>
#include <pdsdata/xtc/TypeId.hh>
#include <psddl_python/DdlWrapper.h>

#include <pdsdata/xtc/DetInfo.hh> // other included packages
namespace Psana {
namespace EvrData {

using namespace boost::python;
using boost::python::api::object;
using boost::shared_ptr;
using std::vector;

extern void createWrappers();

class PulseConfig_Wrapper {
  shared_ptr<PulseConfig> _o;
  PulseConfig* o;
public:
  PulseConfig_Wrapper(shared_ptr<PulseConfig> obj) : _o(obj), o(_o.get()) {}
  PulseConfig_Wrapper(PulseConfig* obj) : o(obj) {}
  uint32_t pulse() const { return o->pulse(); }
  uint32_t _input_control_value() const { return o->_input_control_value(); }
  int16_t bf_trigger() const { return o->bf_trigger(); }
  int16_t bf_set() const { return o->bf_set(); }
  int16_t bf_clear() const { return o->bf_clear(); }
  uint32_t _output_control_value() const { return o->_output_control_value(); }
  uint8_t polarity() const { return o->polarity(); }
  uint8_t map_set_enable() const { return o->map_set_enable(); }
  uint8_t map_reset_enable() const { return o->map_reset_enable(); }
  uint8_t map_trigger_enable() const { return o->map_trigger_enable(); }
  uint32_t prescale() const { return o->prescale(); }
  uint32_t delay() const { return o->delay(); }
  uint32_t width() const { return o->width(); }
  int16_t trigger() const { return o->trigger(); }
  int16_t set() const { return o->set(); }
  int16_t clear() const { return o->clear(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class PulseConfigV3_Wrapper {
  shared_ptr<PulseConfigV3> _o;
  PulseConfigV3* o;
public:
  PulseConfigV3_Wrapper(shared_ptr<PulseConfigV3> obj) : _o(obj), o(_o.get()) {}
  PulseConfigV3_Wrapper(PulseConfigV3* obj) : o(obj) {}
  uint16_t pulseId() const { return o->pulseId(); }
  uint16_t polarity() const { return o->polarity(); }
  uint32_t prescale() const { return o->prescale(); }
  uint32_t delay() const { return o->delay(); }
  uint32_t width() const { return o->width(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class EventCodeV3_Wrapper {
  shared_ptr<EventCodeV3> _o;
  EventCodeV3* o;
public:
  EventCodeV3_Wrapper(shared_ptr<EventCodeV3> obj) : _o(obj), o(_o.get()) {}
  EventCodeV3_Wrapper(EventCodeV3* obj) : o(obj) {}
  uint16_t code() const { return o->code(); }
  uint16_t _u16MaskEventAttr_value() const { return o->_u16MaskEventAttr_value(); }
  uint8_t isReadout() const { return o->isReadout(); }
  uint8_t isTerminator() const { return o->isTerminator(); }
  uint32_t maskTrigger() const { return o->maskTrigger(); }
  uint32_t maskSet() const { return o->maskSet(); }
  uint32_t maskClear() const { return o->maskClear(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class EventCodeV4_Wrapper {
  shared_ptr<EventCodeV4> _o;
  EventCodeV4* o;
public:
  EventCodeV4_Wrapper(shared_ptr<EventCodeV4> obj) : _o(obj), o(_o.get()) {}
  EventCodeV4_Wrapper(EventCodeV4* obj) : o(obj) {}
  uint16_t code() const { return o->code(); }
  uint16_t _u16MaskEventAttr_value() const { return o->_u16MaskEventAttr_value(); }
  uint8_t isReadout() const { return o->isReadout(); }
  uint8_t isTerminator() const { return o->isTerminator(); }
  uint32_t reportDelay() const { return o->reportDelay(); }
  uint32_t reportWidth() const { return o->reportWidth(); }
  uint32_t maskTrigger() const { return o->maskTrigger(); }
  uint32_t maskSet() const { return o->maskSet(); }
  uint32_t maskClear() const { return o->maskClear(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class EventCodeV5_Wrapper {
  shared_ptr<EventCodeV5> _o;
  EventCodeV5* o;
public:
  EventCodeV5_Wrapper(shared_ptr<EventCodeV5> obj) : _o(obj), o(_o.get()) {}
  EventCodeV5_Wrapper(EventCodeV5* obj) : o(obj) {}
  uint16_t code() const { return o->code(); }
  uint8_t isReadout() const { return o->isReadout(); }
  uint8_t isTerminator() const { return o->isTerminator(); }
  uint8_t isLatch() const { return o->isLatch(); }
  uint32_t reportDelay() const { return o->reportDelay(); }
  uint32_t reportWidth() const { return o->reportWidth(); }
  uint32_t maskTrigger() const { return o->maskTrigger(); }
  uint32_t maskSet() const { return o->maskSet(); }
  uint32_t maskClear() const { return o->maskClear(); }
  const char* desc() const { return o->desc(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
  vector<int> desc_shape() const { return o->desc_shape(); }
};

class OutputMap_Wrapper {
  shared_ptr<OutputMap> _o;
  OutputMap* o;
public:
  OutputMap_Wrapper(shared_ptr<OutputMap> obj) : _o(obj), o(_o.get()) {}
  OutputMap_Wrapper(OutputMap* obj) : o(obj) {}
  uint32_t value() const { return o->value(); }
  EvrData::OutputMap::Source source() const { return o->source(); }
  uint8_t source_id() const { return o->source_id(); }
  EvrData::OutputMap::Conn conn() const { return o->conn(); }
  uint8_t conn_id() const { return o->conn_id(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class OutputMapV2_Wrapper {
  shared_ptr<OutputMapV2> _o;
  OutputMapV2* o;
public:
  OutputMapV2_Wrapper(shared_ptr<OutputMapV2> obj) : _o(obj), o(_o.get()) {}
  OutputMapV2_Wrapper(OutputMapV2* obj) : o(obj) {}
  uint32_t value() const { return o->value(); }
  EvrData::OutputMapV2::Source source() const { return o->source(); }
  uint8_t source_id() const { return o->source_id(); }
  EvrData::OutputMapV2::Conn conn() const { return o->conn(); }
  uint8_t conn_id() const { return o->conn_id(); }
  uint8_t module() const { return o->module(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class ConfigV1_Wrapper {
  shared_ptr<ConfigV1> _o;
  ConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_EvrConfig };
  enum { Version = 1 };
  ConfigV1_Wrapper(shared_ptr<ConfigV1> obj) : _o(obj), o(_o.get()) {}
  ConfigV1_Wrapper(ConfigV1* obj) : o(obj) {}
  uint32_t npulses() const { return o->npulses(); }
  uint32_t noutputs() const { return o->noutputs(); }
  vector<EvrData::PulseConfig> pulses() const { VEC_CONVERT(o->pulses(), EvrData::PulseConfig); }
  vector<EvrData::OutputMap> output_maps() const { VEC_CONVERT(o->output_maps(), EvrData::OutputMap); }
};

class ConfigV2_Wrapper {
  shared_ptr<ConfigV2> _o;
  ConfigV2* o;
public:
  enum { TypeId = Pds::TypeId::Id_EvrConfig };
  enum { Version = 2 };
  ConfigV2_Wrapper(shared_ptr<ConfigV2> obj) : _o(obj), o(_o.get()) {}
  ConfigV2_Wrapper(ConfigV2* obj) : o(obj) {}
  uint32_t opcode() const { return o->opcode(); }
  uint32_t npulses() const { return o->npulses(); }
  uint32_t noutputs() const { return o->noutputs(); }
  vector<EvrData::PulseConfig> pulses() const { VEC_CONVERT(o->pulses(), EvrData::PulseConfig); }
  vector<EvrData::OutputMap> output_maps() const { VEC_CONVERT(o->output_maps(), EvrData::OutputMap); }
  EvrData::ConfigV2::BeamCode beam() const { return o->beam(); }
  EvrData::ConfigV2::RateCode rate() const { return o->rate(); }
};

class ConfigV3_Wrapper {
  shared_ptr<ConfigV3> _o;
  ConfigV3* o;
public:
  enum { TypeId = Pds::TypeId::Id_EvrConfig };
  enum { Version = 3 };
  ConfigV3_Wrapper(shared_ptr<ConfigV3> obj) : _o(obj), o(_o.get()) {}
  ConfigV3_Wrapper(ConfigV3* obj) : o(obj) {}
  uint32_t neventcodes() const { return o->neventcodes(); }
  uint32_t npulses() const { return o->npulses(); }
  uint32_t noutputs() const { return o->noutputs(); }
  vector<EvrData::EventCodeV3> eventcodes() const { VEC_CONVERT(o->eventcodes(), EvrData::EventCodeV3); }
  vector<EvrData::PulseConfigV3> pulses() const { VEC_CONVERT(o->pulses(), EvrData::PulseConfigV3); }
  vector<EvrData::OutputMap> output_maps() const { VEC_CONVERT(o->output_maps(), EvrData::OutputMap); }
};

class ConfigV4_Wrapper {
  shared_ptr<ConfigV4> _o;
  ConfigV4* o;
public:
  enum { TypeId = Pds::TypeId::Id_EvrConfig };
  enum { Version = 4 };
  ConfigV4_Wrapper(shared_ptr<ConfigV4> obj) : _o(obj), o(_o.get()) {}
  ConfigV4_Wrapper(ConfigV4* obj) : o(obj) {}
  uint32_t neventcodes() const { return o->neventcodes(); }
  uint32_t npulses() const { return o->npulses(); }
  uint32_t noutputs() const { return o->noutputs(); }
  vector<EvrData::EventCodeV4> eventcodes() const { VEC_CONVERT(o->eventcodes(), EvrData::EventCodeV4); }
  vector<EvrData::PulseConfigV3> pulses() const { VEC_CONVERT(o->pulses(), EvrData::PulseConfigV3); }
  vector<EvrData::OutputMap> output_maps() const { VEC_CONVERT(o->output_maps(), EvrData::OutputMap); }
};

class SequencerEntry_Wrapper {
  shared_ptr<SequencerEntry> _o;
  SequencerEntry* o;
public:
  SequencerEntry_Wrapper(shared_ptr<SequencerEntry> obj) : _o(obj), o(_o.get()) {}
  SequencerEntry_Wrapper(SequencerEntry* obj) : o(obj) {}
  uint32_t delay() const { return o->delay(); }
  uint32_t eventcode() const { return o->eventcode(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class SequencerConfigV1_Wrapper {
  shared_ptr<SequencerConfigV1> _o;
  SequencerConfigV1* o;
public:
  SequencerConfigV1_Wrapper(shared_ptr<SequencerConfigV1> obj) : _o(obj), o(_o.get()) {}
  SequencerConfigV1_Wrapper(SequencerConfigV1* obj) : o(obj) {}
  EvrData::SequencerConfigV1::Source sync_source() const { return o->sync_source(); }
  EvrData::SequencerConfigV1::Source beam_source() const { return o->beam_source(); }
  uint32_t length() const { return o->length(); }
  uint32_t cycles() const { return o->cycles(); }
  vector<EvrData::SequencerEntry> entries() const { VEC_CONVERT(o->entries(), EvrData::SequencerEntry); }
};

class ConfigV5_Wrapper {
  shared_ptr<ConfigV5> _o;
  ConfigV5* o;
public:
  enum { TypeId = Pds::TypeId::Id_EvrConfig };
  enum { Version = 5 };
  ConfigV5_Wrapper(shared_ptr<ConfigV5> obj) : _o(obj), o(_o.get()) {}
  ConfigV5_Wrapper(ConfigV5* obj) : o(obj) {}
  uint32_t neventcodes() const { return o->neventcodes(); }
  uint32_t npulses() const { return o->npulses(); }
  uint32_t noutputs() const { return o->noutputs(); }
  vector<EvrData::EventCodeV5> eventcodes() const { VEC_CONVERT(o->eventcodes(), EvrData::EventCodeV5); }
  vector<EvrData::PulseConfigV3> pulses() const { VEC_CONVERT(o->pulses(), EvrData::PulseConfigV3); }
  vector<EvrData::OutputMap> output_maps() const { VEC_CONVERT(o->output_maps(), EvrData::OutputMap); }
  const SequencerConfigV1_Wrapper seq_config() const { return SequencerConfigV1_Wrapper((SequencerConfigV1*) &o->seq_config()); }
};

class ConfigV6_Wrapper {
  shared_ptr<ConfigV6> _o;
  ConfigV6* o;
public:
  enum { TypeId = Pds::TypeId::Id_EvrConfig };
  enum { Version = 6 };
  ConfigV6_Wrapper(shared_ptr<ConfigV6> obj) : _o(obj), o(_o.get()) {}
  ConfigV6_Wrapper(ConfigV6* obj) : o(obj) {}
  uint32_t neventcodes() const { return o->neventcodes(); }
  uint32_t npulses() const { return o->npulses(); }
  uint32_t noutputs() const { return o->noutputs(); }
  vector<EvrData::EventCodeV5> eventcodes() const { VEC_CONVERT(o->eventcodes(), EvrData::EventCodeV5); }
  vector<EvrData::PulseConfigV3> pulses() const { VEC_CONVERT(o->pulses(), EvrData::PulseConfigV3); }
  vector<EvrData::OutputMapV2> output_maps() const { VEC_CONVERT(o->output_maps(), EvrData::OutputMapV2); }
  const SequencerConfigV1_Wrapper seq_config() const { return SequencerConfigV1_Wrapper((SequencerConfigV1*) &o->seq_config()); }
};

class FIFOEvent_Wrapper {
  shared_ptr<FIFOEvent> _o;
  FIFOEvent* o;
public:
  FIFOEvent_Wrapper(shared_ptr<FIFOEvent> obj) : _o(obj), o(_o.get()) {}
  FIFOEvent_Wrapper(FIFOEvent* obj) : o(obj) {}
  uint32_t timestampHigh() const { return o->timestampHigh(); }
  uint32_t timestampLow() const { return o->timestampLow(); }
  uint32_t eventCode() const { return o->eventCode(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class DataV3_Wrapper {
  shared_ptr<DataV3> _o;
  DataV3* o;
public:
  enum { TypeId = Pds::TypeId::Id_EvrData };
  enum { Version = 3 };
  DataV3_Wrapper(shared_ptr<DataV3> obj) : _o(obj), o(_o.get()) {}
  DataV3_Wrapper(DataV3* obj) : o(obj) {}
  uint32_t numFifoEvents() const { return o->numFifoEvents(); }
  vector<EvrData::FIFOEvent> fifoEvents() const { VEC_CONVERT(o->fifoEvents(), EvrData::FIFOEvent); }
};

class IOChannel_Wrapper {
  shared_ptr<IOChannel> _o;
  IOChannel* o;
public:
  IOChannel_Wrapper(shared_ptr<IOChannel> obj) : _o(obj), o(_o.get()) {}
  IOChannel_Wrapper(IOChannel* obj) : o(obj) {}
  const char* name() const { return o->name(); }
  uint32_t ninfo() const { return o->ninfo(); }
  vector<Pds::DetInfo> infos() const { VEC_CONVERT(o->infos(), Pds::DetInfo); }
  uint32_t _sizeof() const { return o->_sizeof(); }
  vector<int> name_shape() const { return o->name_shape(); }
};

class IOConfigV1_Wrapper {
  shared_ptr<IOConfigV1> _o;
  IOConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_EvrIOConfig };
  enum { Version = 1 };
  IOConfigV1_Wrapper(shared_ptr<IOConfigV1> obj) : _o(obj), o(_o.get()) {}
  IOConfigV1_Wrapper(IOConfigV1* obj) : o(obj) {}
  uint16_t nchannels() const { return o->nchannels(); }
  vector<EvrData::IOChannel> channels() const { VEC_CONVERT(o->channels(), EvrData::IOChannel); }
  EvrData::OutputMap::Conn conn() const { return o->conn(); }
};

  class PulseConfig_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::PulseConfig";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<PulseConfig> result = evt.get(key, foundSrc);
      return result.get() ? object(PulseConfig_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<PulseConfig> result = evt.get(src, key, foundSrc);
      return result.get() ? object(PulseConfig_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<PulseConfig> result = evt.get(source, key, foundSrc);
      return result.get() ? object(PulseConfig_Wrapper(result)) : object();
    }
  };

  class PulseConfigV3_Getter : public Psana::EnvGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::PulseConfigV3";
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<PulseConfigV3> result = store.get(src, 0);
      return result.get() ? object(PulseConfigV3_Wrapper(result)) : object();
    }
  };

  class EventCodeV3_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::EventCodeV3";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<EventCodeV3> result = evt.get(key, foundSrc);
      return result.get() ? object(EventCodeV3_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<EventCodeV3> result = evt.get(src, key, foundSrc);
      return result.get() ? object(EventCodeV3_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<EventCodeV3> result = evt.get(source, key, foundSrc);
      return result.get() ? object(EventCodeV3_Wrapper(result)) : object();
    }
  };

  class EventCodeV4_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::EventCodeV4";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<EventCodeV4> result = evt.get(key, foundSrc);
      return result.get() ? object(EventCodeV4_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<EventCodeV4> result = evt.get(src, key, foundSrc);
      return result.get() ? object(EventCodeV4_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<EventCodeV4> result = evt.get(source, key, foundSrc);
      return result.get() ? object(EventCodeV4_Wrapper(result)) : object();
    }
  };

  class EventCodeV5_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::EventCodeV5";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<EventCodeV5> result = evt.get(key, foundSrc);
      return result.get() ? object(EventCodeV5_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<EventCodeV5> result = evt.get(src, key, foundSrc);
      return result.get() ? object(EventCodeV5_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<EventCodeV5> result = evt.get(source, key, foundSrc);
      return result.get() ? object(EventCodeV5_Wrapper(result)) : object();
    }
  };

  class OutputMap_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::OutputMap";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<OutputMap> result = evt.get(key, foundSrc);
      return result.get() ? object(OutputMap_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<OutputMap> result = evt.get(src, key, foundSrc);
      return result.get() ? object(OutputMap_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<OutputMap> result = evt.get(source, key, foundSrc);
      return result.get() ? object(OutputMap_Wrapper(result)) : object();
    }
  };

  class OutputMapV2_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::OutputMapV2";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<OutputMapV2> result = evt.get(key, foundSrc);
      return result.get() ? object(OutputMapV2_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<OutputMapV2> result = evt.get(src, key, foundSrc);
      return result.get() ? object(OutputMapV2_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<OutputMapV2> result = evt.get(source, key, foundSrc);
      return result.get() ? object(OutputMapV2_Wrapper(result)) : object();
    }
  };

  class ConfigV1_Getter : public Psana::EnvGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::ConfigV1";
    }
    int getVersion() {
      return ConfigV1::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<ConfigV1> result = store.get(src, 0);
      return result.get() ? object(ConfigV1_Wrapper(result)) : object();
    }
  };

  class ConfigV2_Getter : public Psana::EnvGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::ConfigV2";
    }
    int getVersion() {
      return ConfigV2::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<ConfigV2> result = store.get(src, 0);
      return result.get() ? object(ConfigV2_Wrapper(result)) : object();
    }
  };

  class ConfigV3_Getter : public Psana::EnvGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::ConfigV3";
    }
    int getVersion() {
      return ConfigV3::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<ConfigV3> result = store.get(src, 0);
      return result.get() ? object(ConfigV3_Wrapper(result)) : object();
    }
  };

  class ConfigV4_Getter : public Psana::EnvGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::ConfigV4";
    }
    int getVersion() {
      return ConfigV4::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<ConfigV4> result = store.get(src, 0);
      return result.get() ? object(ConfigV4_Wrapper(result)) : object();
    }
  };

  class SequencerEntry_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::SequencerEntry";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<SequencerEntry> result = evt.get(key, foundSrc);
      return result.get() ? object(SequencerEntry_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<SequencerEntry> result = evt.get(src, key, foundSrc);
      return result.get() ? object(SequencerEntry_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<SequencerEntry> result = evt.get(source, key, foundSrc);
      return result.get() ? object(SequencerEntry_Wrapper(result)) : object();
    }
  };

  class SequencerConfigV1_Getter : public Psana::EnvGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::SequencerConfigV1";
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<SequencerConfigV1> result = store.get(src, 0);
      return result.get() ? object(SequencerConfigV1_Wrapper(result)) : object();
    }
  };

  class ConfigV5_Getter : public Psana::EnvGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::ConfigV5";
    }
    int getVersion() {
      return ConfigV5::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<ConfigV5> result = store.get(src, 0);
      return result.get() ? object(ConfigV5_Wrapper(result)) : object();
    }
  };

  class ConfigV6_Getter : public Psana::EnvGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::ConfigV6";
    }
    int getVersion() {
      return ConfigV6::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<ConfigV6> result = store.get(src, 0);
      return result.get() ? object(ConfigV6_Wrapper(result)) : object();
    }
  };

  class FIFOEvent_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::FIFOEvent";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<FIFOEvent> result = evt.get(key, foundSrc);
      return result.get() ? object(FIFOEvent_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<FIFOEvent> result = evt.get(src, key, foundSrc);
      return result.get() ? object(FIFOEvent_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<FIFOEvent> result = evt.get(source, key, foundSrc);
      return result.get() ? object(FIFOEvent_Wrapper(result)) : object();
    }
  };

  class DataV3_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::DataV3";
    }
    int getVersion() {
      return DataV3::Version;
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<DataV3> result = evt.get(key, foundSrc);
      return result.get() ? object(DataV3_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<DataV3> result = evt.get(src, key, foundSrc);
      return result.get() ? object(DataV3_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<DataV3> result = evt.get(source, key, foundSrc);
      return result.get() ? object(DataV3_Wrapper(result)) : object();
    }
  };

  class IOChannel_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::IOChannel";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<IOChannel> result = evt.get(key, foundSrc);
      return result.get() ? object(IOChannel_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<IOChannel> result = evt.get(src, key, foundSrc);
      return result.get() ? object(IOChannel_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<IOChannel> result = evt.get(source, key, foundSrc);
      return result.get() ? object(IOChannel_Wrapper(result)) : object();
    }
  };

  class IOConfigV1_Getter : public Psana::EnvGetter {
  public:
    const char* getTypeName() {
      return "Psana::EvrData::IOConfigV1";
    }
    int getVersion() {
      return IOConfigV1::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<IOConfigV1> result = store.get(src, 0);
      return result.get() ? object(IOConfigV1_Wrapper(result)) : object();
    }
  };
} // namespace EvrData
} // namespace Psana
#endif // PSANA_EVR_DDL_WRAPPER_H
