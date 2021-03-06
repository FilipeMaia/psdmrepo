#ifndef PSDDL_HDF2PSANA_ACQIRIS_DDL_H
#define PSDDL_HDF2PSANA_ACQIRIS_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "psddl_psana/acqiris.ddl.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "PSEvt/Proxy.h"
namespace psddl_hdf2psana {
namespace Acqiris {

namespace ns_VertV1_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::Acqiris::VertV1& psanaobj);
  ~dataset_data();

  double fullScale;
  double offset;
  uint32_t coupling;
  uint32_t bandwidth;

  operator Psana::Acqiris::VertV1() const { return Psana::Acqiris::VertV1(fullScale, offset, coupling, bandwidth); }

};
}

namespace ns_HorizV1_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::Acqiris::HorizV1& psanaobj);
  ~dataset_data();

  double sampInterval;
  double delayTime;
  uint32_t nbrSamples;
  uint32_t nbrSegments;

  operator Psana::Acqiris::HorizV1() const { return Psana::Acqiris::HorizV1(sampInterval, delayTime, nbrSamples, nbrSegments); }

};
}
class Proxy_HorizV1_v0 : public PSEvt::Proxy<Psana::Acqiris::HorizV1> {
public:
  typedef Psana::Acqiris::HorizV1 PsanaType;

  Proxy_HorizV1_v0(hdf5pp::Group group, hsize_t idx) : m_group(group), m_idx(idx) {}
  virtual ~Proxy_HorizV1_v0() {}

protected:

  virtual boost::shared_ptr<PsanaType> getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<PsanaType> m_data;
};
boost::shared_ptr<PSEvt::Proxy<Psana::Acqiris::HorizV1> > make_HorizV1(int version, hdf5pp::Group group, hsize_t idx);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Acqiris::HorizV1& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Acqiris::HorizV1& obj, hdf5pp::Group group, hsize_t chunk_size,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method
void append(const Psana::Acqiris::HorizV1& obj, hdf5pp::Group group, int version = -1);


namespace ns_TrigV1_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::Acqiris::TrigV1& psanaobj);
  ~dataset_data();

  uint32_t coupling;
  uint32_t input;
  uint32_t slope;
  double level;

  operator Psana::Acqiris::TrigV1() const { return Psana::Acqiris::TrigV1(coupling, input, slope, level); }

};
}
class Proxy_TrigV1_v0 : public PSEvt::Proxy<Psana::Acqiris::TrigV1> {
public:
  typedef Psana::Acqiris::TrigV1 PsanaType;

  Proxy_TrigV1_v0(hdf5pp::Group group, hsize_t idx) : m_group(group), m_idx(idx) {}
  virtual ~Proxy_TrigV1_v0() {}

protected:

  virtual boost::shared_ptr<PsanaType> getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<PsanaType> m_data;
};
boost::shared_ptr<PSEvt::Proxy<Psana::Acqiris::TrigV1> > make_TrigV1(int version, hdf5pp::Group group, hsize_t idx);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Acqiris::TrigV1& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Acqiris::TrigV1& obj, hdf5pp::Group group, hsize_t chunk_size,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method
void append(const Psana::Acqiris::TrigV1& obj, hdf5pp::Group group, int version = -1);


namespace ns_ConfigV1_v0 {
struct dataset_config {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_config();
  dataset_config(const Psana::Acqiris::ConfigV1& psanaobj);
  ~dataset_config();

  uint32_t nbrConvertersPerChannel;
  uint32_t channelMask;
  uint32_t nbrBanks;
  uint32_t nbrChannels;


};
}


class ConfigV1_v0 : public Psana::Acqiris::ConfigV1 {
public:
  typedef Psana::Acqiris::ConfigV1 PsanaType;
  ConfigV1_v0() {}
  ConfigV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  virtual ~ConfigV1_v0() {}
  virtual uint32_t nbrConvertersPerChannel() const;
  virtual uint32_t channelMask() const;
  virtual uint32_t nbrBanks() const;
  virtual const Psana::Acqiris::TrigV1& trig() const;
  virtual const Psana::Acqiris::HorizV1& horiz() const;
  virtual ndarray<const Psana::Acqiris::VertV1, 1> vert() const;
  virtual uint32_t nbrChannels() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<Acqiris::ns_ConfigV1_v0::dataset_config> m_ds_config;
  void read_ds_config() const;
  mutable boost::shared_ptr<Acqiris::ns_HorizV1_v0::dataset_data> m_ds_horiz;
  void read_ds_horiz() const;
  mutable Psana::Acqiris::HorizV1 m_ds_storage_horiz;
  mutable boost::shared_ptr<Acqiris::ns_TrigV1_v0::dataset_data> m_ds_trig;
  void read_ds_trig() const;
  mutable Psana::Acqiris::TrigV1 m_ds_storage_trig;
  mutable ndarray<const Psana::Acqiris::VertV1, 1> m_ds_vert;
  void read_ds_vert() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Acqiris::ConfigV1> > make_ConfigV1(int version, hdf5pp::Group group, hsize_t idx);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Acqiris::ConfigV1& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Acqiris::ConfigV1& obj, hdf5pp::Group group, hsize_t chunk_size,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method
void append(const Psana::Acqiris::ConfigV1& obj, hdf5pp::Group group, int version = -1);


namespace ns_TimestampV1_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::Acqiris::TimestampV1& psanaobj);
  ~dataset_data();

  double pos;
  uint64_t value;


};
}
boost::shared_ptr<PSEvt::Proxy<Psana::Acqiris::DataDescV1> > make_DataDescV1(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Acqiris::ConfigV1>& cfg);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Acqiris::DataDescV1& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Acqiris::DataDescV1& obj, hdf5pp::Group group, hsize_t chunk_size,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method
void append(const Psana::Acqiris::DataDescV1& obj, hdf5pp::Group group, int version = -1);


namespace ns_TdcChannel_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::Acqiris::TdcChannel& psanaobj);
  ~dataset_data();

  uint32_t channel;
  uint32_t _mode_int;
  uint16_t slope;
  uint16_t mode;
  double level;

  operator Psana::Acqiris::TdcChannel() const { return Psana::Acqiris::TdcChannel(Psana::Acqiris::TdcChannel::Channel(channel), Psana::Acqiris::TdcChannel::Slope(slope), Psana::Acqiris::TdcChannel::Mode(mode), level); }

};
}
class Proxy_TdcChannel_v0 : public PSEvt::Proxy<Psana::Acqiris::TdcChannel> {
public:
  typedef Psana::Acqiris::TdcChannel PsanaType;

  Proxy_TdcChannel_v0(hdf5pp::Group group, hsize_t idx) : m_group(group), m_idx(idx) {}
  virtual ~Proxy_TdcChannel_v0() {}

protected:

  virtual boost::shared_ptr<PsanaType> getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<PsanaType> m_data;
};
boost::shared_ptr<PSEvt::Proxy<Psana::Acqiris::TdcChannel> > make_TdcChannel(int version, hdf5pp::Group group, hsize_t idx);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Acqiris::TdcChannel& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Acqiris::TdcChannel& obj, hdf5pp::Group group, hsize_t chunk_size,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method
void append(const Psana::Acqiris::TdcChannel& obj, hdf5pp::Group group, int version = -1);


namespace ns_TdcAuxIO_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::Acqiris::TdcAuxIO& psanaobj);
  ~dataset_data();

  uint32_t channel;
  uint32_t mode;
  uint32_t term;

  operator Psana::Acqiris::TdcAuxIO() const { return Psana::Acqiris::TdcAuxIO(Psana::Acqiris::TdcAuxIO::Channel(channel), Psana::Acqiris::TdcAuxIO::Mode(mode), Psana::Acqiris::TdcAuxIO::Termination(term)); }

};
}
class Proxy_TdcAuxIO_v0 : public PSEvt::Proxy<Psana::Acqiris::TdcAuxIO> {
public:
  typedef Psana::Acqiris::TdcAuxIO PsanaType;

  Proxy_TdcAuxIO_v0(hdf5pp::Group group, hsize_t idx) : m_group(group), m_idx(idx) {}
  virtual ~Proxy_TdcAuxIO_v0() {}

protected:

  virtual boost::shared_ptr<PsanaType> getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<PsanaType> m_data;
};
boost::shared_ptr<PSEvt::Proxy<Psana::Acqiris::TdcAuxIO> > make_TdcAuxIO(int version, hdf5pp::Group group, hsize_t idx);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Acqiris::TdcAuxIO& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Acqiris::TdcAuxIO& obj, hdf5pp::Group group, hsize_t chunk_size,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method
void append(const Psana::Acqiris::TdcAuxIO& obj, hdf5pp::Group group, int version = -1);


namespace ns_TdcVetoIO_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::Acqiris::TdcVetoIO& psanaobj);
  ~dataset_data();

  uint32_t channel;
  uint32_t mode;
  uint32_t term;

  operator Psana::Acqiris::TdcVetoIO() const { return Psana::Acqiris::TdcVetoIO(Psana::Acqiris::TdcVetoIO::Mode(mode), Psana::Acqiris::TdcVetoIO::Termination(term)); }

};
}
class Proxy_TdcVetoIO_v0 : public PSEvt::Proxy<Psana::Acqiris::TdcVetoIO> {
public:
  typedef Psana::Acqiris::TdcVetoIO PsanaType;

  Proxy_TdcVetoIO_v0(hdf5pp::Group group, hsize_t idx) : m_group(group), m_idx(idx) {}
  virtual ~Proxy_TdcVetoIO_v0() {}

protected:

  virtual boost::shared_ptr<PsanaType> getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<PsanaType> m_data;
};
boost::shared_ptr<PSEvt::Proxy<Psana::Acqiris::TdcVetoIO> > make_TdcVetoIO(int version, hdf5pp::Group group, hsize_t idx);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Acqiris::TdcVetoIO& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Acqiris::TdcVetoIO& obj, hdf5pp::Group group, hsize_t chunk_size,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method
void append(const Psana::Acqiris::TdcVetoIO& obj, hdf5pp::Group group, int version = -1);



class TdcConfigV1_v0 : public Psana::Acqiris::TdcConfigV1 {
public:
  typedef Psana::Acqiris::TdcConfigV1 PsanaType;
  TdcConfigV1_v0() {}
  TdcConfigV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  virtual ~TdcConfigV1_v0() {}
  virtual ndarray<const Psana::Acqiris::TdcChannel, 1> channels() const;
  virtual ndarray<const Psana::Acqiris::TdcAuxIO, 1> auxio() const;
  virtual const Psana::Acqiris::TdcVetoIO& veto() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<Acqiris::ns_TdcVetoIO_v0::dataset_data> m_ds_veto;
  void read_ds_veto() const;
  mutable Psana::Acqiris::TdcVetoIO m_ds_storage_veto;
  mutable ndarray<const Psana::Acqiris::TdcChannel, 1> m_ds_channel;
  void read_ds_channel() const;
  mutable ndarray<const Psana::Acqiris::TdcAuxIO, 1> m_ds_auxio;
  void read_ds_auxio() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Acqiris::TdcConfigV1> > make_TdcConfigV1(int version, hdf5pp::Group group, hsize_t idx);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Acqiris::TdcConfigV1& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Acqiris::TdcConfigV1& obj, hdf5pp::Group group, hsize_t chunk_size,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method
void append(const Psana::Acqiris::TdcConfigV1& obj, hdf5pp::Group group, int version = -1);


namespace ns_TdcDataV1_Item_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::Acqiris::TdcDataV1_Item& psanaobj);
  ~dataset_data();

  int32_t source;
  uint8_t overflow;
  uint32_t value;

  operator Psana::Acqiris::TdcDataV1_Item() const { return Psana::Acqiris::TdcDataV1_Item(value, Psana::Acqiris::TdcDataV1_Item::Source(source), overflow); }

};
}
class Proxy_TdcDataV1_Item_v0 : public PSEvt::Proxy<Psana::Acqiris::TdcDataV1_Item> {
public:
  typedef Psana::Acqiris::TdcDataV1_Item PsanaType;

  Proxy_TdcDataV1_Item_v0(hdf5pp::Group group, hsize_t idx) : m_group(group), m_idx(idx) {}
  virtual ~Proxy_TdcDataV1_Item_v0() {}

protected:

  virtual boost::shared_ptr<PsanaType> getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<PsanaType> m_data;
};
boost::shared_ptr<PSEvt::Proxy<Psana::Acqiris::TdcDataV1_Item> > make_TdcDataV1_Item(int version, hdf5pp::Group group, hsize_t idx);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Acqiris::TdcDataV1_Item& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Acqiris::TdcDataV1_Item& obj, hdf5pp::Group group, hsize_t chunk_size,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method
void append(const Psana::Acqiris::TdcDataV1_Item& obj, hdf5pp::Group group, int version = -1);



class TdcDataV1_v0 : public Psana::Acqiris::TdcDataV1 {
public:
  typedef Psana::Acqiris::TdcDataV1 PsanaType;
  TdcDataV1_v0() {}
  TdcDataV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  TdcDataV1_v0(const ndarray<const Psana::Acqiris::TdcDataV1_Item, 1>& ds) : m_ds_data(ds) {}
  virtual ~TdcDataV1_v0() {}
  virtual ndarray<const Psana::Acqiris::TdcDataV1_Item, 1> data() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable ndarray<const Psana::Acqiris::TdcDataV1_Item, 1> m_ds_data;
  void read_ds_data() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Acqiris::TdcDataV1> > make_TdcDataV1(int version, hdf5pp::Group group, hsize_t idx);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Acqiris::TdcDataV1& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Acqiris::TdcDataV1& obj, hdf5pp::Group group, hsize_t chunk_size,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method
void append(const Psana::Acqiris::TdcDataV1& obj, hdf5pp::Group group, int version = -1);

} // namespace Acqiris
} // namespace psddl_hdf2psana
#endif // PSDDL_HDF2PSANA_ACQIRIS_DDL_H
