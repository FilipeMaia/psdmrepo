#ifndef PSDDL_HDF2PSANA_EVR_H
#define PSDDL_HDF2PSANA_EVR_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Hand-written supporting types for DDL-HDF5 mapping.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "PSEvt/Proxy.h"
#include "psddl_hdf2psana/ChunkPolicy.h"
#include "psddl_hdf2psana/evr.ddl.h"
#include "psddl_hdf2psana/xtc.h"
#include "psddl_psana/evr.ddl.h"

namespace psddl_hdf2psana {
namespace EvrData {



namespace ns_DataV3_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::EvrData::DataV3& psanaobj);
  ~dataset_data();

  size_t vlen_fifoEvents;
  EvrData::ns_FIFOEvent_v0::dataset_data* fifoEvents;

private:
  dataset_data(const dataset_data& ds);
  dataset_data& operator=(const dataset_data& ds);
};
}


class DataV3_v0 : public Psana::EvrData::DataV3 {
public:
  typedef Psana::EvrData::DataV3 PsanaType;
  DataV3_v0() {}
  DataV3_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  DataV3_v0(const boost::shared_ptr<EvrData::ns_DataV3_v0::dataset_data>& ds) : m_ds_data(ds) {}
  virtual ~DataV3_v0() {}
    uint32_t numFifoEvents() const;

  virtual ndarray<const Psana::EvrData::FIFOEvent, 1> fifoEvents() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<EvrData::ns_DataV3_v0::dataset_data> m_ds_data;
  void read_ds_data() const;
  mutable ndarray<const Psana::EvrData::FIFOEvent, 1> m_ds_storage_data_fifoEvents;
};


void make_datasets_DataV3_v0(const Psana::EvrData::DataV3& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_DataV3_v0(const Psana::EvrData::DataV3* obj, hdf5pp::Group group, long index, bool append);

///  --------- begin Data V4
namespace ns_DataV4_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::EvrData::DataV4& psanaobj);
  ~dataset_data();

  size_t vlen_fifoEvents;
  EvrData::ns_FIFOEvent_v0::dataset_data* fifoEvents;

private:
  dataset_data(const dataset_data& ds);
  dataset_data& operator=(const dataset_data& ds);
};
}


class DataV4_v0 : public Psana::EvrData::DataV4 {
public:
  typedef Psana::EvrData::DataV4 PsanaType;
  DataV4_v0() {}
  DataV4_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  DataV4_v0(const boost::shared_ptr<EvrData::ns_DataV4_v0::dataset_data>& ds) : m_ds_data(ds) {}
  virtual ~DataV4_v0() {}
    uint32_t numFifoEvents() const;

  virtual ndarray<const Psana::EvrData::FIFOEvent, 1> fifoEvents() const;

  virtual uint8_t present(uint8_t) const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<EvrData::ns_DataV4_v0::dataset_data> m_ds_data;
  void read_ds_data() const;
  mutable ndarray<const Psana::EvrData::FIFOEvent, 1> m_ds_storage_data_fifoEvents;
};


void make_datasets_DataV4_v0(const Psana::EvrData::DataV4& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_DataV4_v0(const Psana::EvrData::DataV4* obj, hdf5pp::Group group, long index, bool append);

//// --- end Data V4

namespace ns_IOChannel_v0 {
struct dataset_data {

  enum {NameLength = Psana::EvrData::IOChannel::NameLength};
  enum {MaxInfos = Psana::EvrData::IOChannel::MaxInfos};

  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::EvrData::IOChannel& psanaobj);
  dataset_data(const dataset_data& ds);
  ~dataset_data();

  dataset_data& operator=(const dataset_data& ds);

  char name[NameLength];
  size_t ninfo;
  Pds::ns_DetInfo_v0::dataset_data* infos;

  operator Psana::EvrData::IOChannel() const;

};
}

class Proxy_IOChannel_v0 : public PSEvt::Proxy<Psana::EvrData::IOChannel> {
public:
  typedef Psana::EvrData::IOChannel PsanaType;

  Proxy_IOChannel_v0(hdf5pp::Group group, hsize_t idx) : m_group(group), m_idx(idx) {}
  virtual ~Proxy_IOChannel_v0() {}

protected:

  virtual boost::shared_ptr<PsanaType> getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key);

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<PsanaType> m_data;
};



namespace ns_IOConfigV1_v0 {
struct dataset_config {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_config();
  dataset_config(const Psana::EvrData::IOConfigV1& psanaobj);
  ~dataset_config();

  int32_t conn;
};
}


class IOConfigV1_v0 : public Psana::EvrData::IOConfigV1 {
public:
  typedef Psana::EvrData::IOConfigV1 PsanaType;
  IOConfigV1_v0() {}
  IOConfigV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  virtual ~IOConfigV1_v0() {}
  virtual uint16_t nchannels() const;
  virtual ndarray<const Psana::EvrData::IOChannel, 1> channels() const;
  virtual Psana::EvrData::OutputMap::Conn conn() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;

  mutable boost::shared_ptr<EvrData::ns_IOConfigV1_v0::dataset_config> m_ds_config;
  void read_ds_config() const;

  mutable ndarray<const Psana::EvrData::IOChannel, 1> m_ds_channels;
  void read_ds_channels() const;
};

void make_datasets_IOConfigV1_v0(const Psana::EvrData::IOConfigV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_IOConfigV1_v0(const Psana::EvrData::IOConfigV1* obj, hdf5pp::Group group, long index, bool append);



namespace ns_IOChannelV2_v0 {
struct dataset_data {


  enum {NameLength = Psana::EvrData::IOChannelV2::NameLength};
  enum {MaxInfos = Psana::EvrData::IOChannelV2::MaxInfos};

  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::EvrData::IOChannelV2& psanaobj);
  ~dataset_data();

  EvrData::ns_OutputMapV2_v0::dataset_data output;
  char* name;
  uint32_t ninfo;
  Pds::ns_DetInfo_v0::dataset_data infos[MaxInfos];

  operator Psana::EvrData::IOChannelV2() const;

};
}


namespace ns_IOConfigV2_v0 {
struct dataset_config {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_config();
  dataset_config(const Psana::EvrData::IOConfigV2& psanaobj);
  ~dataset_config();

  uint32_t nchannels;


};
}


class IOConfigV2_v0 : public Psana::EvrData::IOConfigV2 {
public:
  typedef Psana::EvrData::IOConfigV2 PsanaType;
  IOConfigV2_v0() {}
  IOConfigV2_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  virtual ~IOConfigV2_v0() {}
  virtual uint32_t nchannels() const;
  virtual ndarray<const Psana::EvrData::IOChannelV2, 1> channels() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<EvrData::ns_IOConfigV2_v0::dataset_config> m_ds_config;
  void read_ds_config() const;
  mutable ndarray<const Psana::EvrData::IOChannelV2, 1> m_ds_channels;
  void read_ds_channels() const;
};


void make_datasets_IOConfigV2_v0(const Psana::EvrData::IOConfigV2& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);

void store_IOConfigV2_v0(const Psana::EvrData::IOConfigV2* obj, hdf5pp::Group group, long index, bool append);


} // namespace EvrData
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_EVR.DDLM_H
