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
  ~dataset_data();

  size_t vlen_fifoEvents;
  EvrData::ns_FIFOEvent_v0::dataset_data* fifoEvents;

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



namespace ns_IOChannel_v0 {
struct dataset_data {

  enum {NameLength = Psana::EvrData::IOChannel::NameLength};
  enum {MaxInfos = Psana::EvrData::IOChannel::MaxInfos};

  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  ~dataset_data();

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

} // namespace EvrData
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_EVR.DDLM_H
