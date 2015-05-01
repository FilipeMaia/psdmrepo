//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Hand-written supporting types for DDL-HDF5 mapping.
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_hdf2psana/camera.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/Utils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace psddl_hdf2psana {
namespace Camera {

hdf5pp::Type ns_FrameV1_v0_dataset_data_stored_type()
{
  typedef ns_FrameV1_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("width", offsetof(DsType, width), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("height", offsetof(DsType, height), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("depth", offsetof(DsType, depth), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("offset", offsetof(DsType, offset), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_FrameV1_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_FrameV1_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_FrameV1_v0_dataset_data_native_type()
{
  typedef ns_FrameV1_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("width", offsetof(DsType, width), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("height", offsetof(DsType, height), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("depth", offsetof(DsType, depth), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("offset", offsetof(DsType, offset), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

hdf5pp::Type ns_FrameV1_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_FrameV1_v0_dataset_data_native_type();
  return type;
}

ns_FrameV1_v0::dataset_data::dataset_data()
{
}

ns_FrameV1_v0::dataset_data::dataset_data(const Psana::Camera::FrameV1& psanaobj)
  : width(psanaobj.width())
  , height(psanaobj.height())
  , depth(psanaobj.depth())
  , offset(psanaobj.offset())
{
}

ns_FrameV1_v0::dataset_data::~dataset_data()
{
}
uint32_t FrameV1_v0::width() const {
  if (not m_ds_data.get()) read_ds_data();
  return uint32_t(m_ds_data->width);
}
uint32_t FrameV1_v0::height() const {
  if (not m_ds_data.get()) read_ds_data();
  return uint32_t(m_ds_data->height);
}
uint32_t FrameV1_v0::depth() const {
  if (not m_ds_data.get()) read_ds_data();
  return uint32_t(m_ds_data->depth);
}
uint32_t FrameV1_v0::offset() const {
  if (not m_ds_data.get()) read_ds_data();
  return uint32_t(m_ds_data->offset);
}

ndarray<const uint8_t, 1> FrameV1_v0::_int_pixel_data() const {
  if (m_ds_image.empty()) read_ds_image();
  return m_ds_image;
}

ndarray<const uint8_t, 2>
FrameV1_v0::data8() const
{
  if (this->depth() > 8) return ndarray<const uint8_t, 2>();
  if (m_ds_image.empty()) read_ds_image();
  return make_ndarray(m_ds_image.data_ptr(), height(), width());
}

ndarray<const uint16_t, 2>
FrameV1_v0::data16() const
{
  if (this->depth() <= 8) return ndarray<const uint16_t, 2>();
  if (m_ds_image.empty()) read_ds_image();
  boost::shared_ptr<const uint16_t> tptr(m_ds_image.data_ptr(), (const uint16_t*)m_ds_image.data_ptr().get());
  return make_ndarray(tptr, height(), width());
}

/** Number of bytes per pixel. */
uint32_t 
FrameV1_v0::depth_bytes() const
{
  return (depth() + 7) / 8;
}


void FrameV1_v0::read_ds_data() const
{
  m_ds_data = hdf5pp::Utils::readGroup<Camera::ns_FrameV1_v0::dataset_data>(m_group, "data", m_idx);
}

void FrameV1_v0::read_ds_image() const
{
  // Image in HDF5 is stored as rank-2 array of either 8-bit or 16-bit data

  // open dataset and check the type
  hdf5pp::DataSet ds = m_group.openDataSet("image");
  if (ds.type().super().size() == 1) {
    // single-byte
    ndarray<const uint8_t, 2> img = hdf5pp::Utils::readNdarray<uint8_t, 2>(ds, m_idx);
    m_ds_image = make_ndarray(img.data_ptr(), img.size());
  } else {
    // otherwise 16-bit
    ndarray<const uint16_t, 2> img = hdf5pp::Utils::readNdarray<uint16_t, 2>(m_group, "image", m_idx);
    boost::shared_ptr<const uint8_t> tptr(img.data_ptr(), (const uint8_t*)img.data_ptr().get());
    m_ds_image = make_ndarray(tptr, img.size()*2);
  }
}

void make_datasets_FrameV1_v0(const Psana::Camera::FrameV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  {
    hdf5pp::Type dstype = ns_FrameV1_v0::dataset_data::stored_type();
    hdf5pp::Utils::createDataset(group, "data", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }

  {
    // image can be either 8-bit or 16-bit
    hsize_t dims[2];
    hdf5pp::Type basetype;
    if (obj.depth() > 8) {
      const ndarray<const uint16_t, 2>& psana_array = obj.data16();
      std::copy(psana_array.shape(), psana_array.shape()+2, dims);
      basetype = hdf5pp::TypeTraits<uint16_t>::stored_type();
    } else {
      const ndarray<const uint8_t, 2>& psana_array = obj.data8();
      std::copy(psana_array.shape(), psana_array.shape()+2, dims);
      basetype = hdf5pp::TypeTraits<uint8_t>::stored_type();
    }
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(basetype, 2, dims);
    hdf5pp::Utils::createDataset(group, "image", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
}

void store_FrameV1_v0(const Psana::Camera::FrameV1* obj, hdf5pp::Group group, long index, bool append)
{
  if (not obj) {
    if (append) {
      hdf5pp::Utils::resizeDataset(group, "data", index < 0 ? index : index + 1);
      hdf5pp::Utils::resizeDataset(group, "image", index < 0 ? index : index + 1);
    }
    return;
  }

  // store metadata dataset
  if (append) {
    hdf5pp::Utils::storeAt(group, "data", ns_FrameV1_v0::dataset_data(*obj), index);
  } else {
    hdf5pp::Utils::storeScalar(group, "data", ns_FrameV1_v0::dataset_data(*obj));
  }

  // store image, it can be either 8-bit or 16-bit
  if (obj->depth() > 8) {

    ndarray<const uint16_t, 2> img = obj->data16();
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "image", img, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "image", img);
    }

  } else {

    ndarray<const uint8_t, 2> img = obj->data8();
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "image", img, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "image", img);
    }

  }

}

} // namespace Camera
} // namespace psddl_hdf2psana
