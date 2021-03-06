#ifndef PSDDL_HDF2PSANA_ALIAS_DDL_H
#define PSDDL_HDF2PSANA_ALIAS_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "psddl_psana/alias.ddl.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "PSEvt/Proxy.h"
#include "psddl_hdf2psana/ChunkPolicy.h"
#include "pdsdata/xtc/Src.hh"
#include "psddl_hdf2psana/xtc.h"
namespace psddl_hdf2psana {
namespace Alias {

namespace ns_SrcAlias_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::Alias::SrcAlias& psanaobj);
  ~dataset_data();

  Pds::ns_Src_v0::dataset_data src;
  char aliasName[31];

  operator Psana::Alias::SrcAlias() const { return Psana::Alias::SrcAlias(src, aliasName); }

};
}

namespace ns_ConfigV1_v0 {
struct dataset_config {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_config();
  dataset_config(const Psana::Alias::ConfigV1& psanaobj);
  ~dataset_config();

  uint32_t numSrcAlias;


};
}


class ConfigV1_v0 : public Psana::Alias::ConfigV1 {
public:
  typedef Psana::Alias::ConfigV1 PsanaType;
  ConfigV1_v0() {}
  ConfigV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  virtual ~ConfigV1_v0() {}
  virtual uint32_t numSrcAlias() const;
  virtual ndarray<const Psana::Alias::SrcAlias, 1> srcAlias() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<Alias::ns_ConfigV1_v0::dataset_config> m_ds_config;
  void read_ds_config() const;
  mutable ndarray<const Psana::Alias::SrcAlias, 1> m_ds_aliases;
  void read_ds_aliases() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Alias::ConfigV1> > make_ConfigV1(int version, hdf5pp::Group group, hsize_t idx);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Alias::ConfigV1& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Alias::ConfigV1& obj, hdf5pp::Group group, const ChunkPolicy& chunkPolicy,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method at the specified index,
/// negative index means append to the end of dataset. If pointer to object is zero then
/// datsets are extended with zero-filled of default-initialized data.
void store_at(const Psana::Alias::ConfigV1* obj, hdf5pp::Group group, long index = -1, int version = -1);

} // namespace Alias
} // namespace psddl_hdf2psana
#endif // PSDDL_HDF2PSANA_ALIAS_DDL_H
