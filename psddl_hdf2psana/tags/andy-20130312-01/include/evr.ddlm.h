#ifndef PSDDL_HDF2PSANA_EVR_DDLM_H
#define PSDDL_HDF2PSANA_EVR_DDLM_H

#include <boost/shared_ptr.hpp>

#include "hdf5pp/Group.h"
#include "psddl_hdf2psana/evr.ddl.h"
#include "psddl_hdf2psana/xtc.ddlm.h"
#include "psddl_psana/evr.ddl.h"
#include "PSEvt/Proxy.h"

namespace psddl_hdf2psana {
namespace EvrData {

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
