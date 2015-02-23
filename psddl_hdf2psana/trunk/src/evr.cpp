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

namespace {
  const int EvrDataPresentTableSize = 256;
} // local namespace

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

ns_DataV3_v0::dataset_data::dataset_data(const Psana::EvrData::DataV3& psanaobj)
{
  ndarray<const Psana::EvrData::FIFOEvent, 1> fifos = psanaobj.fifoEvents();
  vlen_fifoEvents = fifos.shape()[0];
  fifoEvents = (ns_FIFOEvent_v0::dataset_data*)malloc(sizeof(ns_FIFOEvent_v0::dataset_data)*vlen_fifoEvents);
  for(unsigned i = 0; i != vlen_fifoEvents; ++ i) {
    new (fifoEvents+i) ns_FIFOEvent_v0::dataset_data(fifos[i]);
  }
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


uint32_t DataV3_v0::numFifoEvents() const
{
  if (not m_ds_data) read_ds_data();
  return m_ds_data->vlen_fifoEvents;
}

void make_datasets_DataV3_v0(const Psana::EvrData::DataV3& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  {
    hdf5pp::Type dstype = ns_DataV3_v0::dataset_data::stored_type();
    hdf5pp::Utils::createDataset(group, "data", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
}

void store_DataV3_v0(const Psana::EvrData::DataV3* obj, hdf5pp::Group group, long index, bool append)
{
  if (obj) {
    ns_DataV3_v0::dataset_data ds_data(*obj);
    if (append) {
      hdf5pp::Utils::storeAt(group, "data", ds_data, index);
    } else {
      hdf5pp::Utils::storeScalar(group, "data", ds_data);
    }
  } else if (append) {
    hdf5pp::Utils::resizeDataset(group, "data", index < 0 ? index : index + 1);
  }
}

/// begin DataV4

hdf5pp::Type ns_DataV4_v0_dataset_data_stored_type()
{
  typedef ns_DataV4_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  hdf5pp::VlenType _array_type_fifoEvents = hdf5pp::VlenType::vlenType(hdf5pp::TypeTraits<EvrData::ns_FIFOEvent_v0::dataset_data>::stored_type());
  type.insert("fifoEvents", offsetof(DsType, vlen_fifoEvents), _array_type_fifoEvents);
  return type;
}

hdf5pp::Type ns_DataV4_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_DataV4_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_DataV4_v0_dataset_data_native_type()
{
  typedef ns_DataV4_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  hdf5pp::VlenType _array_type_fifoEvents = hdf5pp::VlenType::vlenType(hdf5pp::TypeTraits<EvrData::ns_FIFOEvent_v0::dataset_data>::native_type());
  type.insert("fifoEvents", offsetof(DsType, vlen_fifoEvents), _array_type_fifoEvents);
  return type;
}

hdf5pp::Type ns_DataV4_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_DataV4_v0_dataset_data_native_type();
  return type;
}

ns_DataV4_v0::dataset_data::dataset_data()
{
  this->vlen_fifoEvents = 0;
  this->fifoEvents = 0;
}

ns_DataV4_v0::dataset_data::dataset_data(const Psana::EvrData::DataV4& psanaobj)
{
  ndarray<const Psana::EvrData::FIFOEvent, 1> fifos = psanaobj.fifoEvents();
  vlen_fifoEvents = fifos.shape()[0];
  fifoEvents = (ns_FIFOEvent_v0::dataset_data*)malloc(sizeof(ns_FIFOEvent_v0::dataset_data)*vlen_fifoEvents);
  for(unsigned i = 0; i != vlen_fifoEvents; ++ i) {
    new (fifoEvents+i) ns_FIFOEvent_v0::dataset_data(fifos[i]);
  }
}

ns_DataV4_v0::dataset_data::~dataset_data()
{
  free(this->fifoEvents);
}

ndarray<const Psana::EvrData::FIFOEvent, 1> DataV4_v0::fifoEvents() const {
  if (not m_ds_data) read_ds_data();
  if (m_ds_storage_data_fifoEvents.empty()) {
    unsigned shape[] = {m_ds_data->vlen_fifoEvents};
    ndarray<Psana::EvrData::FIFOEvent, 1> tmparr(shape);
    std::copy(m_ds_data->fifoEvents, m_ds_data->fifoEvents+m_ds_data->vlen_fifoEvents, tmparr.begin());
    m_ds_storage_data_fifoEvents = tmparr;
  }
  return m_ds_storage_data_fifoEvents;
}

uint8_t DataV4_v0::present(uint8_t opcode) const {
  ndarray<const Psana::EvrData::FIFOEvent, 1> fifoEvents = this->fifoEvents();
  for (unsigned idx = 0; idx < this->numFifoEvents(); idx++) {
    if (fifoEvents[idx].eventCode() == opcode) return 1;
  }
  return 0;
}

void DataV4_v0::read_ds_data() const {
  std::string dsname = "data";
  m_ds_data = hdf5pp::Utils::readGroup<EvrData::ns_DataV4_v0::dataset_data>(m_group, dsname, m_idx);
}


uint32_t DataV4_v0::numFifoEvents() const
{
  if (not m_ds_data) read_ds_data();
  return m_ds_data->vlen_fifoEvents;
}

void make_datasets_DataV4_v0(const Psana::EvrData::DataV4& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  {
    hdf5pp::Type dstype = ns_DataV4_v0::dataset_data::stored_type();
    hdf5pp::Utils::createDataset(group, "data", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);

    // special dataset for present table
    hsize_t dim = EvrDataPresentTableSize;
    hdf5pp::Type dstype2 = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<uint8_t>::stored_type(), dim);
    hdf5pp::Utils::createDataset(group, "present", dstype2, 
                                 chunkPolicy.chunkSize(dstype2), 
                                 chunkPolicy.chunkCacheSize(dstype2), deflate, shuffle);
  }
}

void store_DataV4_v0(const Psana::EvrData::DataV4* obj, hdf5pp::Group group, long index, bool append)
{
  if (obj) {
    ns_DataV4_v0::dataset_data ds_data(*obj);
    // make present table
    ndarray<uint8_t, 1> presentData = make_ndarray<uint8_t>(EvrDataPresentTableSize);
    for (uint16_t opcode16 = 0; opcode16 < EvrDataPresentTableSize; ++opcode16) {
      uint8_t opcode = uint8_t(opcode16);
      presentData[opcode]=obj->present(opcode);
    }
    if (append) {
      hdf5pp::Utils::storeAt(group, "data", ds_data, index);
      hdf5pp::Utils::storeNDArrayAt(group, "present", presentData, index);
    } else {
      hdf5pp::Utils::storeScalar(group, "data", ds_data);
    hdf5pp::Utils::storeNDArray(group, "present", presentData);
    }
  } else if (append) {
    hdf5pp::Utils::resizeDataset(group, "data", index < 0 ? index : index + 1);
    hdf5pp::Utils::resizeDataset(group, "present", index < 0 ? index : index + 1);
  }
}


/// end DataV4

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
  : ninfo(0)
  , infos(0)
{
  name[0] = '\0';
}

ns_IOChannel_v0::dataset_data::dataset_data(const Psana::EvrData::IOChannel& psanaobj)
{
  strncpy(name, psanaobj.name(), NameLength);
  name[NameLength-1] = '\0';

  ndarray<const Pds::DetInfo, 1> dinfos = psanaobj.infos();
  ninfo = dinfos.shape()[0];
  infos = (Pds::ns_DetInfo_v0::dataset_data*)malloc(sizeof(Pds::ns_DetInfo_v0::dataset_data)*ninfo);
  for(unsigned i = 0; i != ninfo; ++ i) {
    new (infos+i) Pds::ns_DetInfo_v0::dataset_data(dinfos[i]);
  }
}

ns_IOChannel_v0::dataset_data::dataset_data(const dataset_data& ds)
{
  strncpy(name, ds.name, NameLength);
  name[NameLength-1] = '\0';

  ninfo = ds.ninfo;
  infos = (Pds::ns_DetInfo_v0::dataset_data*)malloc(sizeof(Pds::ns_DetInfo_v0::dataset_data)*ninfo);
  std::copy(ds.infos, ds.infos+ninfo, infos);
}

ns_IOChannel_v0::dataset_data&
ns_IOChannel_v0::dataset_data::operator=(const dataset_data& ds)
{
  if (this != &ds) {
    free(infos);

    strncpy(name, ds.name, NameLength);
    name[NameLength-1] = '\0';

    ninfo = ds.ninfo;
    infos = (Pds::ns_DetInfo_v0::dataset_data*)malloc(sizeof(Pds::ns_DetInfo_v0::dataset_data)*ninfo);
    std::copy(ds.infos, ds.infos+ninfo, infos);
  }
  return *this;
}

ns_IOChannel_v0::dataset_data::~dataset_data()
{
  free(infos);
}

ns_IOChannel_v0::dataset_data::operator Psana::EvrData::IOChannel() const
{
  Pds::DetInfo dinfos[MaxInfos];
  memset(dinfos,0,MaxInfos*sizeof(Pds::DetInfo));
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

ns_IOConfigV1_v0::dataset_config::dataset_config(const Psana::EvrData::IOConfigV1& psanaobj)
  : conn(psanaobj.conn())
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


void make_datasets_IOConfigV1_v0(const Psana::EvrData::IOConfigV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  {
    hdf5pp::Type dstype = ns_IOConfigV1_v0::dataset_config::stored_type();
    hdf5pp::Utils::createDataset(group, "config", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
  {
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<ns_IOChannel_v0::dataset_data>::stored_type(), obj.channels().shape()[0]);
    hdf5pp::Utils::createDataset(group, "channels", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
}

void store_IOConfigV1_v0(const Psana::EvrData::IOConfigV1* obj, hdf5pp::Group group, long index, bool append)
{
  if (not obj) {
    if (append) {
      hdf5pp::Utils::resizeDataset(group, "config", index < 0 ? index : index + 1);
      hdf5pp::Utils::resizeDataset(group, "channels", index < 0 ? index : index + 1);
    }
    return;
  }

  // convert IOChannel data
  ndarray<const Psana::EvrData::IOChannel, 1> channels = obj->channels();
  ndarray<ns_IOChannel_v0::dataset_data, 1> xchannels(channels.shape());
  for (unsigned i = 0; i != channels.shape()[0]; ++ i) {
    xchannels[i] = ns_IOChannel_v0::dataset_data(channels[i]);
  }

  if (append) {
    hdf5pp::Utils::storeAt(group, "config", ns_IOConfigV1_v0::dataset_config(*obj), index);
    hdf5pp::Utils::storeNDArrayAt(group, "channels", xchannels, index);
  } else {
    hdf5pp::Utils::storeScalar(group, "config", ns_IOConfigV1_v0::dataset_config(*obj));
    hdf5pp::Utils::storeNDArray(group, "channels", xchannels);
  }

}



hdf5pp::Type ns_IOChannelV2_v0_dataset_data_stored_type()
{
  typedef ns_IOChannelV2_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("output", offsetof(DsType, output), hdf5pp::TypeTraits<EvrData::ns_OutputMapV2_v0::dataset_data>::stored_type());
  type.insert("name", offsetof(DsType, name), hdf5pp::TypeTraits<const char*>::stored_type());
  type.insert("ninfo", offsetof(DsType, ninfo), hdf5pp::TypeTraits<uint32_t>::stored_type());
  hsize_t _array_type_infos_shape[] = { 16 };
  hdf5pp::ArrayType _array_type_infos = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<Pds::ns_DetInfo_v0::dataset_data>::stored_type(), 1, _array_type_infos_shape);
  type.insert("infos", offsetof(DsType, infos), _array_type_infos);
  return type;
}

hdf5pp::Type ns_IOChannelV2_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_IOChannelV2_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_IOChannelV2_v0_dataset_data_native_type()
{
  typedef ns_IOChannelV2_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("output", offsetof(DsType, output), hdf5pp::TypeTraits<EvrData::ns_OutputMapV2_v0::dataset_data>::native_type());
  type.insert("name", offsetof(DsType, name), hdf5pp::TypeTraits<const char*>::native_type());
  type.insert("ninfo", offsetof(DsType, ninfo), hdf5pp::TypeTraits<uint32_t>::native_type());
  hsize_t _array_type_infos_shape[] = { ns_IOChannelV2_v0::dataset_data::MaxInfos };
  hdf5pp::ArrayType _array_type_infos = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<Pds::ns_DetInfo_v0::dataset_data>::native_type(), 1, _array_type_infos_shape);
  type.insert("infos", offsetof(DsType, infos), _array_type_infos);
  return type;
}

hdf5pp::Type ns_IOChannelV2_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_IOChannelV2_v0_dataset_data_native_type();
  return type;
}

ns_IOChannelV2_v0::dataset_data::dataset_data()
  : name(NULL)
  , ninfo(0)
{
}

ns_IOChannelV2_v0::dataset_data::dataset_data(const Psana::EvrData::IOChannelV2& psanaobj)
  : output(psanaobj.output())
  , name(NULL)
  , ninfo(psanaobj.ninfo())
{
  name = strdup(psanaobj.name());
  {
    const __typeof__(psanaobj.infos())& arr = psanaobj.infos();
    std::copy(arr.begin(), arr.begin()+MaxInfos, infos);
  }
}

ns_IOChannelV2_v0::dataset_data::~dataset_data()
{
}

ns_IOChannelV2_v0::dataset_data::operator Psana::EvrData::IOChannelV2() const
{
  Pds::DetInfo dinfos[Psana::EvrData::IOChannelV2::MaxInfos];
  std::copy(infos, infos+ninfo, dinfos);
  return Psana::EvrData::IOChannelV2(Psana::EvrData::OutputMapV2(output), name, ninfo, dinfos);
}


hdf5pp::Type ns_IOConfigV2_v0_dataset_config_stored_type()
{
  typedef ns_IOConfigV2_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("nchannels", offsetof(DsType, nchannels), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_IOConfigV2_v0::dataset_config::stored_type()
{
  static hdf5pp::Type type = ns_IOConfigV2_v0_dataset_config_stored_type();
  return type;
}

hdf5pp::Type ns_IOConfigV2_v0_dataset_config_native_type()
{
  typedef ns_IOConfigV2_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("nchannels", offsetof(DsType, nchannels), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

hdf5pp::Type ns_IOConfigV2_v0::dataset_config::native_type()
{
  static hdf5pp::Type type = ns_IOConfigV2_v0_dataset_config_native_type();
  return type;
}

ns_IOConfigV2_v0::dataset_config::dataset_config()
{
}

ns_IOConfigV2_v0::dataset_config::dataset_config(const Psana::EvrData::IOConfigV2& psanaobj)
  : nchannels(psanaobj.nchannels())
{
}

ns_IOConfigV2_v0::dataset_config::~dataset_config()
{
}
uint32_t IOConfigV2_v0::nchannels() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->nchannels);
}
ndarray<const Psana::EvrData::IOChannelV2, 1> IOConfigV2_v0::channels() const {
  if (m_ds_channels.empty()) read_ds_channels();
  return m_ds_channels;
}
void IOConfigV2_v0::read_ds_config() const {
  m_ds_config = hdf5pp::Utils::readGroup<EvrData::ns_IOConfigV2_v0::dataset_config>(m_group, "config", m_idx);
}
void IOConfigV2_v0::read_ds_channels() const {
  ndarray<EvrData::ns_IOChannelV2_v0::dataset_data, 1> arr = hdf5pp::Utils::readNdarray<EvrData::ns_IOChannelV2_v0::dataset_data, 1>(m_group, "channels", m_idx);
  ndarray<Psana::EvrData::IOChannelV2, 1> tmp(arr.shape());
  std::copy(arr.begin(), arr.end(), tmp.begin());
  m_ds_channels = tmp;
}

void make_datasets_IOConfigV2_v0(const Psana::EvrData::IOConfigV2& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  {
    hdf5pp::Type dstype = EvrData::ns_IOConfigV2_v0::dataset_config::stored_type();
    hdf5pp::Utils::createDataset(group, "config", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
  {
    typedef __typeof__(obj.channels()) PsanaArray;
    const PsanaArray& psana_array = obj.channels();
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<EvrData::ns_IOChannelV2_v0::dataset_data>::stored_type(), psana_array.shape()[0]);
    hdf5pp::Utils::createDataset(group, "channels", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
}

void store_IOConfigV2_v0(const Psana::EvrData::IOConfigV2* obj, hdf5pp::Group group, long index, bool append)
{
  if (obj) {
    EvrData::ns_IOConfigV2_v0::dataset_config ds_data(*obj);
    if (append) {
      hdf5pp::Utils::storeAt(group, "config", ds_data, index);
    } else {
      hdf5pp::Utils::storeScalar(group, "config", ds_data);
    }
  } else if (append) {
    hdf5pp::Utils::resizeDataset(group, "config", index < 0 ? index : index + 1);
  }
  if (obj) {
    typedef __typeof__(obj->channels()) PsanaArray;
    typedef ndarray<EvrData::ns_IOChannelV2_v0::dataset_data, 1> HdfArray;
    PsanaArray psana_array = obj->channels();
    HdfArray hdf_array(psana_array.shape());
    HdfArray::iterator out = hdf_array.begin();
    for (PsanaArray::iterator it = psana_array.begin(); it != psana_array.end(); ++ it, ++ out) {
      *out = EvrData::ns_IOChannelV2_v0::dataset_data(*it);
    }
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "channels", hdf_array, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "channels", hdf_array);
    }
  } else if (append) {
    hdf5pp::Utils::resizeDataset(group, "channels", index < 0 ? index : index + 1);
  }

}

} // namespace EvrData
} // namespace psddl_hdf2psana
