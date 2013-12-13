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
#include "psddl_hdf2psana/fccd.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/VlenType.h"
#include "hdf5pp/Utils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  template <typename T, typename U>
  void ndarray2array(const ndarray<T, 1>& src, U* dst, size_t size) {
    size_t i;
    for (i = 0; i < src.shape()[0]; ++ i) dst[i] = src[i];
    for (; i < size; ++ i) dst[i] = U(0);
  }

}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace psddl_hdf2psana {
namespace FCCD {


hdf5pp::Type ns_FccdConfigV2_v0_dataset_config_stored_type()
{
  typedef ns_FccdConfigV2_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("outputMode", offsetof(DsType, outputMode), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("ccdEnable", offsetof(DsType, ccdEnable), hdf5pp::TypeTraits<uint8_t>::stored_type());
  type.insert("focusMode", offsetof(DsType, focusMode), hdf5pp::TypeTraits<uint8_t>::stored_type());
  type.insert("exposureTime", offsetof(DsType, exposureTime), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("width", offsetof(DsType, width), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("height", offsetof(DsType, height), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("trimmedWidth", offsetof(DsType, trimmedWidth), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("trimmedHeight", offsetof(DsType, trimmedHeight), hdf5pp::TypeTraits<uint32_t>::stored_type());
  for (int i = 0; i != Psana::FCCD::FccdConfigV2::NVoltages; ++ i) {
    std::string name = "dacVoltage" + boost::lexical_cast<std::string>(i+1);
    type.insert(name.c_str(), offsetof(DsType, dacVoltage)+i*sizeof(float), hdf5pp::TypeTraits<float>::stored_type());
  }
  for (int i = 0; i != Psana::FCCD::FccdConfigV2::NWaveforms; ++ i) {
    std::string name = "waveform" + boost::lexical_cast<std::string>(i);
    type.insert(name.c_str(), offsetof(DsType, waveform)+i*sizeof(uint16_t), hdf5pp::TypeTraits<uint16_t>::stored_type());
  }
  return type;
}

hdf5pp::Type ns_FccdConfigV2_v0::dataset_config::stored_type()
{
  static hdf5pp::Type type = ns_FccdConfigV2_v0_dataset_config_stored_type();
  return type;
}

hdf5pp::Type ns_FccdConfigV2_v0_dataset_config_native_type()
{
  typedef ns_FccdConfigV2_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("outputMode", offsetof(DsType, outputMode), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("ccdEnable", offsetof(DsType, ccdEnable), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("focusMode", offsetof(DsType, focusMode), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("exposureTime", offsetof(DsType, exposureTime), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("width", offsetof(DsType, width), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("height", offsetof(DsType, height), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("trimmedWidth", offsetof(DsType, trimmedWidth), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("trimmedHeight", offsetof(DsType, trimmedHeight), hdf5pp::TypeTraits<uint32_t>::native_type());
  for (int i = 0; i != Psana::FCCD::FccdConfigV2::NVoltages; ++ i) {
    std::string name = "dacVoltage" + boost::lexical_cast<std::string>(i+1);
    type.insert(name.c_str(), offsetof(DsType, dacVoltage)+i*sizeof(float), hdf5pp::TypeTraits<float>::native_type());
  }
  for (int i = 0; i != Psana::FCCD::FccdConfigV2::NWaveforms; ++ i) {
    std::string name = "waveform" + boost::lexical_cast<std::string>(i);
    type.insert(name.c_str(), offsetof(DsType, waveform)+i*sizeof(uint16_t), hdf5pp::TypeTraits<uint16_t>::native_type());
  }
  return type;
}

hdf5pp::Type ns_FccdConfigV2_v0::dataset_config::native_type()
{
  static hdf5pp::Type type = ns_FccdConfigV2_v0_dataset_config_native_type();
  return type;
}
ns_FccdConfigV2_v0::dataset_config::dataset_config()
{
}

ns_FccdConfigV2_v0::dataset_config::dataset_config(const Psana::FCCD::FccdConfigV2& psanaobj)
  : outputMode(psanaobj.outputMode())
  , ccdEnable(psanaobj.ccdEnable())
  , focusMode(psanaobj.focusMode())
  , exposureTime(psanaobj.exposureTime())
  , width(psanaobj.width())
  , height(psanaobj.height())
  , trimmedWidth(psanaobj.trimmedWidth())
  , trimmedHeight(psanaobj.trimmedHeight())
{
  ndarray2array(psanaobj.dacVoltages(), dacVoltage, Psana::FCCD::FccdConfigV2::NVoltages);
  ndarray2array(psanaobj.waveforms(), waveform, Psana::FCCD::FccdConfigV2::NWaveforms);
}

ns_FccdConfigV2_v0::dataset_config::~dataset_config()
{
}

uint16_t FccdConfigV2_v0::outputMode() const {
  if (not m_ds_config) read_ds_config();
  return uint16_t(m_ds_config->outputMode);
}
uint8_t FccdConfigV2_v0::ccdEnable() const {
  if (not m_ds_config) read_ds_config();
  return uint8_t(m_ds_config->ccdEnable);
}
uint8_t FccdConfigV2_v0::focusMode() const {
  if (not m_ds_config) read_ds_config();
  return uint8_t(m_ds_config->focusMode);
}
uint32_t FccdConfigV2_v0::exposureTime() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->exposureTime);
}
ndarray<const float, 1> FccdConfigV2_v0::dacVoltages() const {
  if (not m_ds_config) read_ds_config();
  boost::shared_ptr<float> ptr(m_ds_config, m_ds_config->dacVoltage);
  return make_ndarray<float>(ptr, Psana::FCCD::FccdConfigV2::NVoltages);
}
ndarray<const uint16_t, 1> FccdConfigV2_v0::waveforms() const {
  if (not m_ds_config) read_ds_config();
  boost::shared_ptr<uint16_t> ptr(m_ds_config, m_ds_config->waveform);
  return make_ndarray<uint16_t>(ptr, Psana::FCCD::FccdConfigV2::NWaveforms);
}
uint32_t FccdConfigV2_v0::width() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->width);
}
uint32_t FccdConfigV2_v0::height() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->height);
}
uint32_t FccdConfigV2_v0::trimmedWidth() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->trimmedWidth);
}
uint32_t FccdConfigV2_v0::trimmedHeight() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->trimmedHeight);
}
void FccdConfigV2_v0::read_ds_config() const {
  m_ds_config = hdf5pp::Utils::readGroup<FCCD::ns_FccdConfigV2_v0::dataset_config>(m_group, "config", m_idx);
}

void make_datasets_FccdConfigV2_v0(const Psana::FCCD::FccdConfigV2& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  {
    hdf5pp::Type dstype = ns_FccdConfigV2_v0::dataset_config::stored_type();
    hdf5pp::Utils::createDataset(group, "config", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
}

void store_FccdConfigV2_v0(const Psana::FCCD::FccdConfigV2* obj, hdf5pp::Group group, long index, bool append)
{
  if (append) {
    if (obj) {
      hdf5pp::Utils::storeAt(group, "config", ns_FccdConfigV2_v0::dataset_config(*obj), index);
    } else {
      hdf5pp::Utils::resizeDataset(group, "config", index < 0 ? index : index + 1);
    }
  } else {
    hdf5pp::Utils::storeScalar(group, "config", ns_FccdConfigV2_v0::dataset_config(*obj));
  }
}


} // namespace FCCD
} // namespace psddl_hdf2psana
