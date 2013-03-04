#include "psddl_hdf2psana/xtc.ddlm.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/Utils.h"
#include "PSEvt/DataProxy.h"

namespace Pds {

hdf5pp::Type ns_ClockTime_v0_dataset_data_stored_type()
{
  typedef ns_ClockTime_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("seconds", offsetof(DsType, seconds), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("nanoseconds", offsetof(DsType, nanoseconds), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_ClockTime_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_ClockTime_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_ClockTime_v0_dataset_data_native_type()
{
  typedef ns_ClockTime_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("seconds", offsetof(DsType, seconds), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("nanoseconds", offsetof(DsType, nanoseconds), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

hdf5pp::Type ns_ClockTime_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_ClockTime_v0_dataset_data_native_type();
  return type;
}

} // namespace Pds
