//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class cspad2x2...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_hdf2psana/cspad2x2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/Utils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psddl_hdf2psana {
namespace CsPad2x2 {

hdf5pp::Type
ns_ElementV1_v0_dataset_element_stored_type()
{
  typedef ns_ElementV1_v0::dataset_element DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("virtual_channel", offsetof(DsType, virtual_channel), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("lane", offsetof(DsType, lane), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("tid", offsetof(DsType, tid), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("acq_count", offsetof(DsType, acq_count), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("op_code", offsetof(DsType, op_code), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("quad", offsetof(DsType, quad), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("seq_count", offsetof(DsType, seq_count), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("ticks", offsetof(DsType, ticks), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("fiducials", offsetof(DsType, fiducials), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("frame_type", offsetof(DsType, frame_type), hdf5pp::TypeTraits<uint32_t>::stored_type());
  hsize_t _array_type_sb_temp_shape[] = { 4 };
  hdf5pp::ArrayType _array_type_sb_temp = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<uint16_t>::stored_type(), 1, _array_type_sb_temp_shape);
  type.insert("sb_temp", offsetof(DsType, sb_temp), _array_type_sb_temp);
  return type;
}

hdf5pp::Type
ns_ElementV1_v0::dataset_element::stored_type()
{
  static hdf5pp::Type type = ns_ElementV1_v0_dataset_element_stored_type();
  return type;
}

hdf5pp::Type
ns_ElementV1_v0_dataset_element_native_type()
{
  typedef ns_ElementV1_v0::dataset_element DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("virtual_channel", offsetof(DsType, virtual_channel), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("lane", offsetof(DsType, lane), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("tid", offsetof(DsType, tid), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("acq_count", offsetof(DsType, acq_count), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("op_code", offsetof(DsType, op_code), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("quad", offsetof(DsType, quad), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("seq_count", offsetof(DsType, seq_count), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("ticks", offsetof(DsType, ticks), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("fiducials", offsetof(DsType, fiducials), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("frame_type", offsetof(DsType, frame_type), hdf5pp::TypeTraits<uint32_t>::native_type());
  hsize_t _array_type_sb_temp_shape[] = { 4 };
  hdf5pp::ArrayType _array_type_sb_temp = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<uint16_t>::native_type(), 1, _array_type_sb_temp_shape);
  type.insert("sb_temp", offsetof(DsType, sb_temp), _array_type_sb_temp);
  return type;
}

hdf5pp::Type
ns_ElementV1_v0::dataset_element::native_type()
{
  static hdf5pp::Type type = ns_ElementV1_v0_dataset_element_native_type();
  return type;
}

ns_ElementV1_v0::dataset_element::dataset_element()
{
}

ns_ElementV1_v0::dataset_element::dataset_element(const Psana::CsPad2x2::ElementV1& psanaobj)
  : virtual_channel(psanaobj.virtual_channel())
  , lane(psanaobj.lane())
  , tid(psanaobj.tid())
  , acq_count(psanaobj.acq_count())
  , op_code(psanaobj.op_code())
  , quad(psanaobj.quad())
  , seq_count(psanaobj.seq_count())
  , ticks(psanaobj.ticks())
  , fiducials(psanaobj.fiducials())
  , frame_type(psanaobj.frame_type())
{
  ndarray<const uint16_t, 1> sb_temp = psanaobj.sb_temp();
  std::copy(sb_temp.begin(), sb_temp.end(), this->sb_temp);
}

uint32_t
ElementV1_v0::virtual_channel() const {
  if (not m_ds_element) read_ds_element();
  return uint32_t(m_ds_element->virtual_channel);
}

uint32_t
ElementV1_v0::lane() const {
  if (not m_ds_element) read_ds_element();
  return uint32_t(m_ds_element->lane);
}

uint32_t
ElementV1_v0::tid() const {
  if (not m_ds_element) read_ds_element();
  return uint32_t(m_ds_element->tid);
}

uint32_t
ElementV1_v0::acq_count() const {
  if (not m_ds_element) read_ds_element();
  return uint32_t(m_ds_element->acq_count);
}

uint32_t
ElementV1_v0::op_code() const {
  if (not m_ds_element) read_ds_element();
  return uint32_t(m_ds_element->op_code);
}

uint32_t
ElementV1_v0::quad() const {
  if (not m_ds_element) read_ds_element();
  return uint32_t(m_ds_element->quad);
}

uint32_t
ElementV1_v0::seq_count() const {
  if (not m_ds_element) read_ds_element();
  return uint32_t(m_ds_element->seq_count);
}

uint32_t
ElementV1_v0::ticks() const {
  if (not m_ds_element) read_ds_element();
  return uint32_t(m_ds_element->ticks);
}

uint32_t
ElementV1_v0::fiducials() const {
  if (not m_ds_element) read_ds_element();
  return uint32_t(m_ds_element->fiducials);
}

ndarray<const uint16_t, 1>
ElementV1_v0::sb_temp() const {
  if (not m_ds_element) read_ds_element();
  boost::shared_ptr<uint16_t> ptr(m_ds_element, m_ds_element->sb_temp);
  return make_ndarray(ptr, Nsbtemp);
}

uint32_t
ElementV1_v0::frame_type() const {
  if (not m_ds_element) read_ds_element();
  return uint32_t(m_ds_element->frame_type);
}

ndarray<const int16_t, 3>
ElementV1_v0::data() const {
  if (m_ds_data.empty()) read_ds_data();
  return m_ds_data;
}

float
ElementV1_v0::common_mode(uint32_t section) const{
  if (m_ds_cm.empty()) read_ds_cm();
  return m_ds_cm[section];
}

void
ElementV1_v0::read_ds_element() const {
  m_ds_element = hdf5pp::Utils::readGroup<CsPad2x2::ns_ElementV1_v0::dataset_element>(m_group, "element", m_idx);
}

void
ElementV1_v0::read_ds_data() const {
  m_ds_data = hdf5pp::Utils::readNdarray<int16_t, 3>(m_group, "data", m_idx);
}

void
ElementV1_v0::read_ds_cm() const {
  if (m_group.hasChild("common_mode")) {
    m_ds_cm = hdf5pp::Utils::readNdarray<float, 1>(m_group, "common_mode", m_idx);
  } else {
    // if dataset is missing make all-zeros array
    m_ds_cm = make_ndarray<float>(2);
    std::fill(m_ds_cm.begin(), m_ds_cm.end(), 0.f);
  }
}

void make_datasets_ElementV1_v0(const Psana::CsPad2x2::ElementV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  {
    hdf5pp::Type dstype = hdf5pp::TypeTraits<ns_ElementV1_v0::dataset_element>::stored_type();
    hdf5pp::Utils::createDataset(group, "element", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
  {
    hsize_t dims[3] = {Psana::CsPad2x2::ColumnsPerASIC, Psana::CsPad2x2::MaxRowsPerASIC*2, 2};
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<int16_t>::stored_type(), 3, dims);
    hdf5pp::Utils::createDataset(group, "data", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
  {
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<float>::stored_type(), 2);
    hdf5pp::Utils::createDataset(group, "common_mode", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
}

void store_ElementV1_v0(const Psana::CsPad2x2::ElementV1* obj, hdf5pp::Group group, long index, bool append)
{
  if (not obj) {
    if (append) {
      hdf5pp::Utils::resizeDataset(group, "element", index < 0 ? index : index + 1);
      hdf5pp::Utils::resizeDataset(group, "data", index < 0 ? index : index + 1);
      hdf5pp::Utils::resizeDataset(group, "common_mode", index < 0 ? index : index + 1);
    }
    return;
  }

  {
    ns_ElementV1_v0::dataset_element data(*obj);
    if (append) {
      hdf5pp::Utils::storeAt(group, "element", data, index);
    } else {
      hdf5pp::Utils::storeScalar(group, "element", data);
    }
  }
  {
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "data", obj->data(), index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "data", obj->data());
    }
  }
  {
    ndarray<float, 1> data = make_ndarray<float>(2);
    data[0] = obj->common_mode(0);
    data[1] = obj->common_mode(1);
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "common_mode", data, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "common_mode", data);
    }
  }
}

} // namespace CsPad2x2
} // namespace psddl_hdf2psana
