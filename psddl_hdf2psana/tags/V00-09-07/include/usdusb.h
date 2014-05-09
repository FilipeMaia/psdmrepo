#ifndef PSDDL_HDF2PSANA_USDUSB_H
#define PSDDL_HDF2PSANA_USDUSB_H 1

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

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "psddl_hdf2psana/ChunkPolicy.h"
#include "psddl_psana/usdusb.ddl.h"

namespace psddl_hdf2psana {
namespace UsdUsb {

// ===============================================================
//      UsbUsd::DataV1 schema version 0
// ===============================================================
namespace ns_DataV1_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::UsdUsb::DataV1& psanaobj);
  ~dataset_data();

  int32_t e_count[4];    // data in v0 are stored as uint32_t but interface requires int32_t
  uint16_t analog_in[4];
  uint32_t timestamp;
  uint8_t status[4];
  uint8_t digital_in;

};
}


class DataV1_v0 : public Psana::UsdUsb::DataV1 {
public:
  typedef Psana::UsdUsb::DataV1 PsanaType;
  DataV1_v0() {}
  DataV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  DataV1_v0(const boost::shared_ptr<UsdUsb::ns_DataV1_v0::dataset_data>& ds) : m_ds_data(ds) {}
  virtual ~DataV1_v0() {}
  virtual uint8_t digital_in() const;
  virtual uint32_t timestamp() const;
  virtual ndarray<const uint8_t, 1> status() const;

  virtual ndarray<const uint16_t, 1> analog_in() const;
  /** Return lower 24 bits of _count array as signed integer values. */
  ndarray<const int32_t, 1> encoder_count() const;

private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<UsdUsb::ns_DataV1_v0::dataset_data> m_ds_data;
  void read_ds_data() const;
};

void make_datasets_DataV1_v0(const Psana::UsdUsb::DataV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_DataV1_v0(const Psana::UsdUsb::DataV1* obj, hdf5pp::Group group, long index, bool append);

} // namespace UsdUsb
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_USDUSB_H
