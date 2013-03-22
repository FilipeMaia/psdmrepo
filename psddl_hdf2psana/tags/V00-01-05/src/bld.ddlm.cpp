#include "psddl_hdf2psana/bld.ddlm.h"


#include <boost/make_shared.hpp>

#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "hdf5pp/Utils.h"
#include "psddl_hdf2psana/lusi.ddl.h"
#include "psddl_hdf2psana/pulnix.ddl.h"

namespace psddl_hdf2psana {
namespace Bld {

hdf5pp::Type
ns_BldDataPimV1_v0_dataset_data_stored_type()
{
  typedef psddl_hdf2psana::Bld::ns_BldDataPimV1_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>() ;
  type.insert( "camConfig", offsetof(DsType, camConfig), hdf5pp::TypeTraits<Pulnix::ns_TM6740ConfigV2_v0::dataset_config>::stored_type() ) ;
  type.insert( "pimConfig", offsetof(DsType, pimConfig), hdf5pp::TypeTraits<Lusi::ns_PimImageConfigV1_v0::dataset_config>::stored_type() ) ;
  type.insert( "frame", offsetof(DsType, frame), hdf5pp::TypeTraits<Camera::ns_FrameV1_v0::dataset_data>::stored_type() ) ;
  return type ;
}

hdf5pp::Type
ns_BldDataPimV1_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_BldDataPimV1_v0_dataset_data_stored_type() ;
  return type ;
}


hdf5pp::Type
ns_BldDataPimV1_v0_dataset_data_native_type()
{
  typedef psddl_hdf2psana::Bld::ns_BldDataPimV1_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>() ;
  type.insert( "camConfig", offsetof(DsType, camConfig), hdf5pp::TypeTraits<Pulnix::ns_TM6740ConfigV2_v0::dataset_config>::native_type() ) ;
  type.insert( "pimConfig", offsetof(DsType, pimConfig), hdf5pp::TypeTraits<Lusi::ns_PimImageConfigV1_v0::dataset_config>::native_type() ) ;
  type.insert( "frame", offsetof(DsType, frame), hdf5pp::TypeTraits<Camera::ns_FrameV1_v0::dataset_data>::native_type() ) ;
  return type ;
}

hdf5pp::Type
ns_BldDataPimV1_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_BldDataPimV1_v0_dataset_data_native_type() ;
  return type ;
}

const Psana::Pulnix::TM6740ConfigV2&
BldDataPimV1_v0::camConfig() const
{
  if (not m_storage_camConfig) {
    if (not m_ds_data) read_ds_data();
    boost::shared_ptr<Pulnix::ns_TM6740ConfigV2_v0::dataset_config> ds(m_ds_data, &m_ds_data->camConfig);
    m_storage_camConfig = boost::make_shared<Pulnix::TM6740ConfigV2_v0>(ds);
  }

  return *m_storage_camConfig;
}

const Psana::Lusi::PimImageConfigV1&
BldDataPimV1_v0::pimConfig() const
{
  if (not m_ds_data) read_ds_data();
  m_storage_pimConfig = Psana::Lusi::PimImageConfigV1(m_ds_data->pimConfig);
  return m_storage_pimConfig;
}

const Psana::Camera::FrameV1&
BldDataPimV1_v0::frame() const
{
  if (not m_storage_frame) {
    if (not m_ds_data) read_ds_data();
    boost::shared_ptr<Camera::ns_FrameV1_v0::dataset_data> ds(m_ds_data, &m_ds_data->frame);
    m_storage_frame = boost::make_shared<Camera::FrameV1_v0>(ds, m_group, m_idx);
  }

  return *m_storage_frame;
}

void
BldDataPimV1_v0::read_ds_data() const
{
  m_ds_data = hdf5pp::Utils::readGroup<ns_BldDataPimV1_v0::dataset_data>(m_group, "data", m_idx);
}

} // namespace Bld
} // namespace psddl_hdf2psana
