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
#include "psddl_hdf2psana/evr.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/Utils.h"
#include "hdf5pp/VlenType.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace psddl_hdf2psana {
namespace EvrData {


hdf5pp::Type ns_DataV3_v0_dataset_data_stored_type()
{
  typedef ns_DataV3_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  hdf5pp::VlenType _array_type_fifoEvents = hdf5pp::VlenType::vlenType(hdf5pp::TypeTraits<EvrData::ns_FIFOEvent_v0::dataset_data>::stored_type());
  type.insert("fifoEvents", offsetof(DsType, vlen_fifoEvents), _array_type_fifoEvents);
  return type;
}

hdf5pp::Type ns_DataV3_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_DataV3_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_DataV3_v0_dataset_data_native_type()
{
  typedef ns_DataV3_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  hdf5pp::VlenType _array_type_fifoEvents = hdf5pp::VlenType::vlenType(hdf5pp::TypeTraits<EvrData::ns_FIFOEvent_v0::dataset_data>::native_type());
  type.insert("fifoEvents", offsetof(DsType, vlen_fifoEvents), _array_type_fifoEvents);
  return type;
}

hdf5pp::Type ns_DataV3_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_DataV3_v0_dataset_data_native_type();
  return type;
}
ns_DataV3_v0::dataset_data::dataset_data()
{
  this->vlen_fifoEvents = 0;
  this->fifoEvents = 0;
}
ns_DataV3_v0::dataset_data::~dataset_data()
{
  free(this->fifoEvents);
}

ndarray<const Psana::EvrData::FIFOEvent, 1> DataV3_v0::fifoEvents() const {
  if (not m_ds_data) read_ds_data();
  if (m_ds_storage_data_fifoEvents.empty()) {
    unsigned shape[] = {m_ds_data->vlen_fifoEvents};
    ndarray<Psana::EvrData::FIFOEvent, 1> tmparr(shape);
    std::copy(m_ds_data->fifoEvents, m_ds_data->fifoEvents+m_ds_data->vlen_fifoEvents, tmparr.begin());
    m_ds_storage_data_fifoEvents = tmparr;
  }
  return m_ds_storage_data_fifoEvents;
}

void DataV3_v0::read_ds_data() const {
  // dataset name for EvrDataV3 has changed at some point from "evrData" to "data",
  // we need to try both here
  std::string dsname = "data";
  if (not m_group.hasChild(dsname)) dsname = "evrData";
  m_ds_data = hdf5pp::Utils::readGroup<EvrData::ns_DataV3_v0::dataset_data>(m_group, dsname, m_idx);
}


void store_DataV3_v0(const Psana::EvrData::DataV3& obj, hdf5pp::Group group, bool append)
{
    
}


hdf5pp::Type ns_IOChannel_v0_dataset_data_stored_type()
{
  typedef ns_IOChannel_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("name", offsetof(DsType, name), hdf5pp::TypeTraits<const char*>::stored_type());

  // variable-size type for info array
  hdf5pp::Type infoType = hdf5pp::VlenType::vlenType ( hdf5pp::TypeTraits<Pds::ns_DetInfo_v0::dataset_data>::stored_type() );

  type.insert("info", offsetof(DsType, ninfo), infoType);
  return type;
}

hdf5pp::Type ns_IOChannel_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_IOChannel_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_IOChannel_v0_dataset_data_native_type()
{
  typedef ns_IOChannel_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("name", offsetof(DsType, name), hdf5pp::TypeTraits<const char*>::native_type());

  // variable-size type for info array
  hdf5pp::Type infoType = hdf5pp::VlenType::vlenType ( hdf5pp::TypeTraits<Pds::ns_DetInfo_v0::dataset_data>::native_type() );

  type.insert("info", offsetof(DsType, ninfo), infoType);
  return type;
}

hdf5pp::Type ns_IOChannel_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_IOChannel_v0_dataset_data_native_type();
  return type;
}
ns_IOChannel_v0::dataset_data::dataset_data()
{
}
ns_IOChannel_v0::dataset_data::~dataset_data()
{
}

ns_IOChannel_v0::dataset_data::operator Psana::EvrData::IOChannel() const
{
  Pds::DetInfo dinfos[MaxInfos];
  std::copy(infos, infos+ninfo, dinfos);
  return Psana::EvrData::IOChannel(name, ninfo, dinfos);
}

boost::shared_ptr<Psana::EvrData::IOChannel>
Proxy_IOChannel_v0::getTypedImpl(PSEvt::ProxyDictI* dict, const Pds::Src& source, const std::string& key)
{
  if (not m_data) {
    boost::shared_ptr<EvrData::ns_IOChannel_v0::dataset_data> ds_data = hdf5pp::Utils::readGroup<EvrData::ns_IOChannel_v0::dataset_data>(m_group, "data", m_idx);
    m_data = boost::make_shared<PsanaType>(*ds_data);
  }

  return m_data;
}




hdf5pp::Type ns_IOConfigV1_v0_dataset_config_stored_type()
{
  typedef ns_IOConfigV1_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  hdf5pp::EnumType<int32_t> _enum_type_conn = hdf5pp::EnumType<int32_t>::enumType();
  _enum_type_conn.insert("FrontPanel", Psana::EvrData::OutputMap::FrontPanel);
  _enum_type_conn.insert("UnivIO", Psana::EvrData::OutputMap::UnivIO);
  type.insert("conn", offsetof(DsType, conn), _enum_type_conn);
  return type;
}

hdf5pp::Type ns_IOConfigV1_v0::dataset_config::stored_type()
{
  static hdf5pp::Type type = ns_IOConfigV1_v0_dataset_config_stored_type();
  return type;
}

hdf5pp::Type ns_IOConfigV1_v0_dataset_config_native_type()
{
  typedef ns_IOConfigV1_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  hdf5pp::EnumType<int32_t> _enum_type_conn = hdf5pp::EnumType<int32_t>::enumType();
  _enum_type_conn.insert("FrontPanel", Psana::EvrData::OutputMap::FrontPanel);
  _enum_type_conn.insert("UnivIO", Psana::EvrData::OutputMap::UnivIO);
  type.insert("conn", offsetof(DsType, conn), _enum_type_conn);
  return type;
}

hdf5pp::Type ns_IOConfigV1_v0::dataset_config::native_type()
{
  static hdf5pp::Type type = ns_IOConfigV1_v0_dataset_config_native_type();
  return type;
}

ns_IOConfigV1_v0::dataset_config::dataset_config()
{
}

ns_IOConfigV1_v0::dataset_config::~dataset_config()
{
}



uint16_t IOConfigV1_v0::nchannels() const {
  if (not m_ds_config.get()) read_ds_channels();
  return m_ds_channels.shape()[0];
}

ndarray<const Psana::EvrData::IOChannel, 1> IOConfigV1_v0::channels() const {
  if (m_ds_channels.empty()) read_ds_channels();
  return m_ds_channels;
}

Psana::EvrData::OutputMap::Conn IOConfigV1_v0::conn() const {
  if (not m_ds_config.get()) read_ds_config();
  return Psana::EvrData::OutputMap::Conn(m_ds_config->conn);
}

void IOConfigV1_v0::read_ds_config() const {
  m_ds_config = hdf5pp::Utils::readGroup<EvrData::ns_IOConfigV1_v0::dataset_config>(m_group, "config", m_idx);
}

void IOConfigV1_v0::read_ds_channels() const {
  ndarray<EvrData::ns_IOChannel_v0::dataset_data, 1> arr = hdf5pp::Utils::readNdarray<EvrData::ns_IOChannel_v0::dataset_data, 1>(m_group, "channels", m_idx);
  ndarray<Psana::EvrData::IOChannel, 1> tmp(arr.shape());
  std::copy(arr.begin(), arr.end(), tmp.begin());
  m_ds_channels = tmp;
}

void store_IOConfigV1_v0(const Psana::EvrData::IOConfigV1& obj, hdf5pp::Group group, bool append)
{
    
}


uint32_t
DataV3_v0::numFifoEvents() const
{
  if (not m_ds_data) read_ds_data();
  return m_ds_data->vlen_fifoEvents;
}

} // namespace EvrData
} // namespace psddl_hdf2psana
