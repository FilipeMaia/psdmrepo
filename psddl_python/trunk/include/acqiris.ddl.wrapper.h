/* Do not edit this file, as it is auto-generated */

#ifndef PSANA_ACQIRIS_DDL_WRAPPER_H
#define PSANA_ACQIRIS_DDL_WRAPPER_H 1

#include <vector>
#include <ndarray/ndarray.h>
#include <pdsdata/xtc/TypeId.hh>
#include <psddl_python/DdlWrapper.h>

namespace Psana {
namespace Acqiris {

using namespace boost::python;
using boost::python::api::object;
using boost::shared_ptr;
using std::vector;

extern void createWrappers();

class VertV1_Wrapper {
  shared_ptr<VertV1> _o;
  VertV1* o;
public:
  enum { Version = 1 };
  VertV1_Wrapper(shared_ptr<VertV1> obj) : _o(obj), o(_o.get()) {}
  VertV1_Wrapper(VertV1* obj) : o(obj) {}
  double fullScale() const { return o->fullScale(); }
  double offset() const { return o->offset(); }
  uint32_t coupling() const { return o->coupling(); }
  uint32_t bandwidth() const { return o->bandwidth(); }
  double slope() const { return o->slope(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class HorizV1_Wrapper {
  shared_ptr<HorizV1> _o;
  HorizV1* o;
public:
  enum { Version = 1 };
  HorizV1_Wrapper(shared_ptr<HorizV1> obj) : _o(obj), o(_o.get()) {}
  HorizV1_Wrapper(HorizV1* obj) : o(obj) {}
  double sampInterval() const { return o->sampInterval(); }
  double delayTime() const { return o->delayTime(); }
  uint32_t nbrSamples() const { return o->nbrSamples(); }
  uint32_t nbrSegments() const { return o->nbrSegments(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class TrigV1_Wrapper {
  shared_ptr<TrigV1> _o;
  TrigV1* o;
public:
  enum { Version = 1 };
  TrigV1_Wrapper(shared_ptr<TrigV1> obj) : _o(obj), o(_o.get()) {}
  TrigV1_Wrapper(TrigV1* obj) : o(obj) {}
  uint32_t coupling() const { return o->coupling(); }
  uint32_t input() const { return o->input(); }
  uint32_t slope() const { return o->slope(); }
  double level() const { return o->level(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class ConfigV1_Wrapper {
  shared_ptr<ConfigV1> _o;
  ConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_AcqConfig };
  enum { Version = 1 };
  ConfigV1_Wrapper(shared_ptr<ConfigV1> obj) : _o(obj), o(_o.get()) {}
  ConfigV1_Wrapper(ConfigV1* obj) : o(obj) {}
  uint32_t nbrConvertersPerChannel() const { return o->nbrConvertersPerChannel(); }
  uint32_t channelMask() const { return o->channelMask(); }
  uint32_t nbrBanks() const { return o->nbrBanks(); }
  const TrigV1_Wrapper trig() const { return TrigV1_Wrapper((TrigV1*) &o->trig()); }
  const HorizV1_Wrapper horiz() const { return HorizV1_Wrapper((HorizV1*) &o->horiz()); }
  vector<Acqiris::VertV1> vert() const { VEC_CONVERT(o->vert(), Acqiris::VertV1); }
  uint32_t nbrChannels() const { return o->nbrChannels(); }
};

class TimestampV1_Wrapper {
  shared_ptr<TimestampV1> _o;
  TimestampV1* o;
public:
  enum { Version = 1 };
  TimestampV1_Wrapper(shared_ptr<TimestampV1> obj) : _o(obj), o(_o.get()) {}
  TimestampV1_Wrapper(TimestampV1* obj) : o(obj) {}
  double pos() const { return o->pos(); }
  uint32_t timeStampLo() const { return o->timeStampLo(); }
  uint32_t timeStampHi() const { return o->timeStampHi(); }
  uint64_t value() const { return o->value(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};
class ConfigV1;

class DataDescV1Elem_Wrapper {
  shared_ptr<DataDescV1Elem> _o;
  DataDescV1Elem* o;
public:
  enum { Version = 1 };
  DataDescV1Elem_Wrapper(shared_ptr<DataDescV1Elem> obj) : _o(obj), o(_o.get()) {}
  DataDescV1Elem_Wrapper(DataDescV1Elem* obj) : o(obj) {}
  uint32_t nbrSamplesInSeg() const { return o->nbrSamplesInSeg(); }
  uint32_t indexFirstPoint() const { return o->indexFirstPoint(); }
  uint32_t nbrSegments() const { return o->nbrSegments(); }
  vector<Acqiris::TimestampV1> timestamp() const { VEC_CONVERT(o->timestamp(), Acqiris::TimestampV1); }
  PyObject* waveforms() const { ND_CONVERT(o->waveforms(), int16_t, 2); }
};
class ConfigV1;

class DataDescV1_Wrapper {
  shared_ptr<DataDescV1> _o;
  DataDescV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_AcqWaveform };
  enum { Version = 1 };
  DataDescV1_Wrapper(shared_ptr<DataDescV1> obj) : _o(obj), o(_o.get()) {}
  DataDescV1_Wrapper(DataDescV1* obj) : o(obj) {}
  const DataDescV1Elem_Wrapper data(uint32_t i0) const { return DataDescV1Elem_Wrapper((DataDescV1Elem*) &o->data(i0)); }
  vector<int> data_shape() const { return o->data_shape(); }
  boost::python::list data_list() { boost::python::list l; const int n = data_shape()[0]; for (int i = 0; i < n; i++) l.append(data(i)); return l; }
};

class TdcChannel_Wrapper {
  shared_ptr<TdcChannel> _o;
  TdcChannel* o;
public:
  TdcChannel_Wrapper(shared_ptr<TdcChannel> obj) : _o(obj), o(_o.get()) {}
  TdcChannel_Wrapper(TdcChannel* obj) : o(obj) {}
  uint32_t _channel_int() const { return o->_channel_int(); }
  uint32_t _mode_int() const { return o->_mode_int(); }
  Acqiris::TdcChannel::Slope slope() const { return o->slope(); }
  Acqiris::TdcChannel::Mode mode() const { return o->mode(); }
  double level() const { return o->level(); }
  Acqiris::TdcChannel::Channel channel() const { return o->channel(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class TdcAuxIO_Wrapper {
  shared_ptr<TdcAuxIO> _o;
  TdcAuxIO* o;
public:
  TdcAuxIO_Wrapper(shared_ptr<TdcAuxIO> obj) : _o(obj), o(_o.get()) {}
  TdcAuxIO_Wrapper(TdcAuxIO* obj) : o(obj) {}
  uint32_t channel_int() const { return o->channel_int(); }
  uint32_t signal_int() const { return o->signal_int(); }
  uint32_t qualifier_int() const { return o->qualifier_int(); }
  Acqiris::TdcAuxIO::Channel channel() const { return o->channel(); }
  Acqiris::TdcAuxIO::Mode mode() const { return o->mode(); }
  Acqiris::TdcAuxIO::Termination term() const { return o->term(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class TdcVetoIO_Wrapper {
  shared_ptr<TdcVetoIO> _o;
  TdcVetoIO* o;
public:
  TdcVetoIO_Wrapper(shared_ptr<TdcVetoIO> obj) : _o(obj), o(_o.get()) {}
  TdcVetoIO_Wrapper(TdcVetoIO* obj) : o(obj) {}
  uint32_t signal_int() const { return o->signal_int(); }
  uint32_t qualifier_int() const { return o->qualifier_int(); }
  Acqiris::TdcVetoIO::Channel channel() const { return o->channel(); }
  Acqiris::TdcVetoIO::Mode mode() const { return o->mode(); }
  Acqiris::TdcVetoIO::Termination term() const { return o->term(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class TdcConfigV1_Wrapper {
  shared_ptr<TdcConfigV1> _o;
  TdcConfigV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_AcqTdcConfig };
  enum { Version = 1 };
  TdcConfigV1_Wrapper(shared_ptr<TdcConfigV1> obj) : _o(obj), o(_o.get()) {}
  TdcConfigV1_Wrapper(TdcConfigV1* obj) : o(obj) {}
  vector<Acqiris::TdcChannel> channels() const { VEC_CONVERT(o->channels(), Acqiris::TdcChannel); }
  vector<Acqiris::TdcAuxIO> auxio() const { VEC_CONVERT(o->auxio(), Acqiris::TdcAuxIO); }
  const TdcVetoIO_Wrapper veto() const { return TdcVetoIO_Wrapper((TdcVetoIO*) &o->veto()); }
};

class TdcDataV1_Item_Wrapper {
  shared_ptr<TdcDataV1_Item> _o;
  TdcDataV1_Item* o;
public:
  TdcDataV1_Item_Wrapper(shared_ptr<TdcDataV1_Item> obj) : _o(obj), o(_o.get()) {}
  TdcDataV1_Item_Wrapper(TdcDataV1_Item* obj) : o(obj) {}
  uint32_t value() const { return o->value(); }
  uint32_t bf_val_() const { return o->bf_val_(); }
  Acqiris::TdcDataV1_Item::Source source() const { return o->source(); }
  uint8_t bf_ofv_() const { return o->bf_ofv_(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class TdcDataV1Common_Wrapper {
  shared_ptr<TdcDataV1Common> _o;
  TdcDataV1Common* o;
public:
  TdcDataV1Common_Wrapper(shared_ptr<TdcDataV1Common> obj) : _o(obj), o(_o.get()) {}
  TdcDataV1Common_Wrapper(TdcDataV1Common* obj) : o(obj) {}
  uint32_t nhits() const { return o->nhits(); }
  uint8_t overflow() const { return o->overflow(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class TdcDataV1Channel_Wrapper {
  shared_ptr<TdcDataV1Channel> _o;
  TdcDataV1Channel* o;
public:
  TdcDataV1Channel_Wrapper(shared_ptr<TdcDataV1Channel> obj) : _o(obj), o(_o.get()) {}
  TdcDataV1Channel_Wrapper(TdcDataV1Channel* obj) : o(obj) {}
  uint32_t ticks() const { return o->ticks(); }
  uint8_t overflow() const { return o->overflow(); }
  double time() const { return o->time(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class TdcDataV1Marker_Wrapper {
  shared_ptr<TdcDataV1Marker> _o;
  TdcDataV1Marker* o;
public:
  TdcDataV1Marker_Wrapper(shared_ptr<TdcDataV1Marker> obj) : _o(obj), o(_o.get()) {}
  TdcDataV1Marker_Wrapper(TdcDataV1Marker* obj) : o(obj) {}
  Acqiris::TdcDataV1Marker::Type type() const { return o->type(); }
  uint32_t _sizeof() const { return o->_sizeof(); }
};

class TdcDataV1_Wrapper {
  shared_ptr<TdcDataV1> _o;
  TdcDataV1* o;
public:
  enum { TypeId = Pds::TypeId::Id_AcqTdcData };
  enum { Version = 1 };
  TdcDataV1_Wrapper(shared_ptr<TdcDataV1> obj) : _o(obj), o(_o.get()) {}
  TdcDataV1_Wrapper(TdcDataV1* obj) : o(obj) {}
  vector<Acqiris::TdcDataV1_Item> data() const { VEC_CONVERT(o->data(), Acqiris::TdcDataV1_Item); }
};

  class VertV1_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::VertV1";
    }
    int getVersion() {
      return VertV1::Version;
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<VertV1> result = evt.get(key, foundSrc);
      return result.get() ? object(VertV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<VertV1> result = evt.get(src, key, foundSrc);
      return result.get() ? object(VertV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<VertV1> result = evt.get(source, key, foundSrc);
      return result.get() ? object(VertV1_Wrapper(result)) : object();
    }
  };

  class HorizV1_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::HorizV1";
    }
    int getVersion() {
      return HorizV1::Version;
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<HorizV1> result = evt.get(key, foundSrc);
      return result.get() ? object(HorizV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<HorizV1> result = evt.get(src, key, foundSrc);
      return result.get() ? object(HorizV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<HorizV1> result = evt.get(source, key, foundSrc);
      return result.get() ? object(HorizV1_Wrapper(result)) : object();
    }
  };

  class TrigV1_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::TrigV1";
    }
    int getVersion() {
      return TrigV1::Version;
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TrigV1> result = evt.get(key, foundSrc);
      return result.get() ? object(TrigV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TrigV1> result = evt.get(src, key, foundSrc);
      return result.get() ? object(TrigV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TrigV1> result = evt.get(source, key, foundSrc);
      return result.get() ? object(TrigV1_Wrapper(result)) : object();
    }
  };

  class ConfigV1_Getter : public Psana::EnvGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::ConfigV1";
    }
    int getVersion() {
      return ConfigV1::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<ConfigV1> result = store.get(src, 0);
      return result.get() ? object(ConfigV1_Wrapper(result)) : object();
    }
  };

  class TimestampV1_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::TimestampV1";
    }
    int getVersion() {
      return TimestampV1::Version;
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TimestampV1> result = evt.get(key, foundSrc);
      return result.get() ? object(TimestampV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TimestampV1> result = evt.get(src, key, foundSrc);
      return result.get() ? object(TimestampV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TimestampV1> result = evt.get(source, key, foundSrc);
      return result.get() ? object(TimestampV1_Wrapper(result)) : object();
    }
  };

  class DataDescV1Elem_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::DataDescV1Elem";
    }
    int getVersion() {
      return DataDescV1Elem::Version;
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<DataDescV1Elem> result = evt.get(key, foundSrc);
      return result.get() ? object(DataDescV1Elem_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<DataDescV1Elem> result = evt.get(src, key, foundSrc);
      return result.get() ? object(DataDescV1Elem_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<DataDescV1Elem> result = evt.get(source, key, foundSrc);
      return result.get() ? object(DataDescV1Elem_Wrapper(result)) : object();
    }
  };

  class DataDescV1_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::DataDescV1";
    }
    int getVersion() {
      return DataDescV1::Version;
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<DataDescV1> result = evt.get(key, foundSrc);
      return result.get() ? object(DataDescV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<DataDescV1> result = evt.get(src, key, foundSrc);
      return result.get() ? object(DataDescV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<DataDescV1> result = evt.get(source, key, foundSrc);
      return result.get() ? object(DataDescV1_Wrapper(result)) : object();
    }
  };

  class TdcChannel_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::TdcChannel";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcChannel> result = evt.get(key, foundSrc);
      return result.get() ? object(TdcChannel_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcChannel> result = evt.get(src, key, foundSrc);
      return result.get() ? object(TdcChannel_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcChannel> result = evt.get(source, key, foundSrc);
      return result.get() ? object(TdcChannel_Wrapper(result)) : object();
    }
  };

  class TdcAuxIO_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::TdcAuxIO";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcAuxIO> result = evt.get(key, foundSrc);
      return result.get() ? object(TdcAuxIO_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcAuxIO> result = evt.get(src, key, foundSrc);
      return result.get() ? object(TdcAuxIO_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcAuxIO> result = evt.get(source, key, foundSrc);
      return result.get() ? object(TdcAuxIO_Wrapper(result)) : object();
    }
  };

  class TdcVetoIO_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::TdcVetoIO";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcVetoIO> result = evt.get(key, foundSrc);
      return result.get() ? object(TdcVetoIO_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcVetoIO> result = evt.get(src, key, foundSrc);
      return result.get() ? object(TdcVetoIO_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcVetoIO> result = evt.get(source, key, foundSrc);
      return result.get() ? object(TdcVetoIO_Wrapper(result)) : object();
    }
  };

  class TdcConfigV1_Getter : public Psana::EnvGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::TdcConfigV1";
    }
    int getVersion() {
      return TdcConfigV1::Version;
    }
    object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src, Pds::Src* foundSrc=0) {
      boost::shared_ptr<TdcConfigV1> result = store.get(src, 0);
      return result.get() ? object(TdcConfigV1_Wrapper(result)) : object();
    }
  };

  class TdcDataV1_Item_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::TdcDataV1_Item";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1_Item> result = evt.get(key, foundSrc);
      return result.get() ? object(TdcDataV1_Item_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1_Item> result = evt.get(src, key, foundSrc);
      return result.get() ? object(TdcDataV1_Item_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1_Item> result = evt.get(source, key, foundSrc);
      return result.get() ? object(TdcDataV1_Item_Wrapper(result)) : object();
    }
  };

  class TdcDataV1Common_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::TdcDataV1Common";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1Common> result = evt.get(key, foundSrc);
      return result.get() ? object(TdcDataV1Common_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1Common> result = evt.get(src, key, foundSrc);
      return result.get() ? object(TdcDataV1Common_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1Common> result = evt.get(source, key, foundSrc);
      return result.get() ? object(TdcDataV1Common_Wrapper(result)) : object();
    }
  };

  class TdcDataV1Channel_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::TdcDataV1Channel";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1Channel> result = evt.get(key, foundSrc);
      return result.get() ? object(TdcDataV1Channel_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1Channel> result = evt.get(src, key, foundSrc);
      return result.get() ? object(TdcDataV1Channel_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1Channel> result = evt.get(source, key, foundSrc);
      return result.get() ? object(TdcDataV1Channel_Wrapper(result)) : object();
    }
  };

  class TdcDataV1Marker_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::TdcDataV1Marker";
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1Marker> result = evt.get(key, foundSrc);
      return result.get() ? object(TdcDataV1Marker_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1Marker> result = evt.get(src, key, foundSrc);
      return result.get() ? object(TdcDataV1Marker_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1Marker> result = evt.get(source, key, foundSrc);
      return result.get() ? object(TdcDataV1Marker_Wrapper(result)) : object();
    }
  };

  class TdcDataV1_Getter : public Psana::EvtGetter {
  public:
    const char* getTypeName() {
      return "Psana::Acqiris::TdcDataV1";
    }
    int getVersion() {
      return TdcDataV1::Version;
    }
    object get(PSEvt::Event& evt, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1> result = evt.get(key, foundSrc);
      return result.get() ? object(TdcDataV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, Pds::Src& src, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1> result = evt.get(src, key, foundSrc);
      return result.get() ? object(TdcDataV1_Wrapper(result)) : object();
    }
    object get(PSEvt::Event& evt, PSEvt::Source& source, const std::string& key=std::string(), Pds::Src* foundSrc=0) {
      shared_ptr<TdcDataV1> result = evt.get(source, key, foundSrc);
      return result.get() ? object(TdcDataV1_Wrapper(result)) : object();
    }
  };
} // namespace Acqiris
} // namespace Psana
#endif // PSANA_ACQIRIS_DDL_WRAPPER_H
