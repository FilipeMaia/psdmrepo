#include "psddl_hdf2psana/usdusb.ddlm.h"

#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/Utils.h"

namespace psddl_hdf2psana {
namespace UsdUsb {


hdf5pp::Type ns_DataV1_v0_dataset_data_stored_type()
{
  typedef ns_DataV1_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("encoder_count", offsetof(DsType, e_count), hdf5pp::TypeTraits<int32_t>::stored_type());
  type.insert("analog_in", offsetof(DsType, analog_in), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("timestamp", offsetof(DsType, timestamp), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("digital_in", offsetof(DsType, digital_in), hdf5pp::TypeTraits<uint8_t>::stored_type());
  return type;
}

hdf5pp::Type ns_DataV1_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_DataV1_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_DataV1_v0_dataset_data_native_type()
{
  typedef ns_DataV1_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("encoder_count", offsetof(DsType, e_count), hdf5pp::TypeTraits<int32_t>::native_type(), 4);
  type.insert("analog_in", offsetof(DsType, analog_in), hdf5pp::TypeTraits<uint16_t>::native_type(), 4);
  type.insert("timestamp", offsetof(DsType, timestamp), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("digital_in", offsetof(DsType, digital_in), hdf5pp::TypeTraits<uint8_t>::native_type());
  return type;
}

hdf5pp::Type
ns_DataV1_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_DataV1_v0_dataset_data_native_type();
  return type;
}

ns_DataV1_v0::dataset_data::dataset_data()
{
}

ns_DataV1_v0::dataset_data::~dataset_data()
{
}

uint8_t
DataV1_v0::digital_in() const
{
  if (not m_ds_data.get()) read_ds_data();
  return uint8_t(m_ds_data->digital_in);
}

uint32_t
DataV1_v0::timestamp() const
{
  if (not m_ds_data.get()) read_ds_data();
  return uint32_t(m_ds_data->timestamp);
}

ndarray<const uint8_t, 1>
DataV1_v0::status() const
{
  ndarray<uint8_t, 1> arr = make_ndarray<uint8_t>(4);
  std::fill_n(arr.begin(), 4, uint8_t(0));
  return arr;
}

ndarray<const uint16_t, 1>
DataV1_v0::analog_in() const
{
  if (not m_ds_data.get()) read_ds_data();
  boost::shared_ptr<uint16_t> ptr(m_ds_data, m_ds_data->analog_in);
  return make_ndarray(ptr, Analog_Inputs);
}

ndarray<const int32_t, 1>
DataV1_v0::encoder_count() const
{
  if (not m_ds_data.get()) read_ds_data();
  boost::shared_ptr<int32_t> ptr(m_ds_data, m_ds_data->e_count);
  return make_ndarray(ptr, Encoder_Inputs);
}

void
DataV1_v0::read_ds_data() const
{
  m_ds_data = hdf5pp::Utils::readGroup<UsdUsb::ns_DataV1_v0::dataset_data>(m_group, "data", m_idx);
}

} // namespace UsdUsb
} // namespace psddl_hdf2psana
