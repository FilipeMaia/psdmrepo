#ifndef PSDDL_HDF2PSANA_CAMERA_H
#define PSDDL_HDF2PSANA_CAMERA_H

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
#include "psddl_psana/camera.ddl.h"
#include "psddl_hdf2psana/ChunkPolicy.h"

namespace psddl_hdf2psana {
namespace Camera {


namespace ns_FrameV1_v0 {
struct dataset_data {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data();
  dataset_data(const Psana::Camera::FrameV1& psanaobj);
  ~dataset_data();

  uint32_t width;
  uint32_t height;
  uint32_t depth;
  uint32_t offset;

};
}


class FrameV1_v0 : public Psana::Camera::FrameV1 {
public:
  typedef Psana::Camera::FrameV1 PsanaType;

  FrameV1_v0() {}
  FrameV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}

  // very specia constructor needed by BldDataPimV1_v0
  FrameV1_v0(const boost::shared_ptr<Camera::ns_FrameV1_v0::dataset_data>& ds_data,
      hdf5pp::Group group, hsize_t idx) : m_group(group), m_idx(idx), m_ds_data(ds_data) {}

  virtual ~FrameV1_v0() {}

  virtual uint32_t width() const;
  virtual uint32_t height() const;
  virtual uint32_t depth() const;
  virtual uint32_t offset() const;
  virtual ndarray<const uint8_t, 1> _int_pixel_data() const;
  /** Returns pixel data array when stored data type is 8-bit (depth() is less than 9).
                If data type is 16-bit then empty array is returned, use data16() method in this case. */
  ndarray<const uint8_t, 2> data8() const;

  /** Returns pixel data array when stored data type is 16-bit (depth() is greater than 8).
                If data type is 8-bit then empty array is returned, use data8() method in this case. */
  ndarray<const uint16_t, 2> data16() const;

  /** Number of bytes per pixel. */
  virtual uint32_t depth_bytes() const;

private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;

  mutable boost::shared_ptr<Camera::ns_FrameV1_v0::dataset_data> m_ds_data;
  void read_ds_data() const;

  mutable ndarray<const uint8_t, 1> m_ds_image;
  void read_ds_image() const;
};

void make_datasets_FrameV1_v0(const Psana::Camera::FrameV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_FrameV1_v0(const Psana::Camera::FrameV1* obj, hdf5pp::Group group, long index, bool append);

} // namespace Camera
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_CAMERA_H
