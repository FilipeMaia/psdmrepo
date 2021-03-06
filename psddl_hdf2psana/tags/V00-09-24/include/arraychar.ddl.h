#ifndef PSDDL_HDF2PSANA_ARRAYCHAR_DDL_H
#define PSDDL_HDF2PSANA_ARRAYCHAR_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "psddl_psana/arraychar.ddl.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "PSEvt/Proxy.h"
#include "psddl_hdf2psana/ChunkPolicy.h"
namespace psddl_hdf2psana {
namespace Arraychar {

namespace ns_DataV1_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::Arraychar::DataV1& psanaobj);
  ~dataset_data();

  uint64_t numChars;
  size_t vlen_data;
  uint8_t* data;


private:
  dataset_data(const dataset_data&);
  dataset_data& operator=(const dataset_data&);
};
}


class DataV1_v0 : public Psana::Arraychar::DataV1 {
public:
  typedef Psana::Arraychar::DataV1 PsanaType;
  DataV1_v0() {}
  DataV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  DataV1_v0(const boost::shared_ptr<Arraychar::ns_DataV1_v0::dataset_data>& ds) : m_ds_data(ds) {}
  virtual ~DataV1_v0() {}
  virtual uint64_t numChars() const;
  virtual ndarray<const uint8_t, 1> data() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<Arraychar::ns_DataV1_v0::dataset_data> m_ds_data;
  void read_ds_data() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::Arraychar::DataV1> > make_DataV1(int version, hdf5pp::Group group, hsize_t idx);

/// Store object as a single instance (scalar dataset) inside specified group.
void store(const Psana::Arraychar::DataV1& obj, hdf5pp::Group group, int version = -1);
/// Create container (rank=1) datasets for storing objects of specified type.
void make_datasets(const Psana::Arraychar::DataV1& obj, hdf5pp::Group group, const ChunkPolicy& chunkPolicy,
                   int deflate, bool shuffle, int version = -1);
/// Add one more object to the containers created by previous method at the specified index,
/// negative index means append to the end of dataset. If pointer to object is zero then
/// datsets are extended with zero-filled of default-initialized data.
void store_at(const Psana::Arraychar::DataV1* obj, hdf5pp::Group group, long index = -1, int version = -1);

} // namespace Arraychar
} // namespace psddl_hdf2psana
#endif // PSDDL_HDF2PSANA_ARRAYCHAR_DDL_H
