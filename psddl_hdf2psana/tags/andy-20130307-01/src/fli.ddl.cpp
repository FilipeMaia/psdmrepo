
// *** Do not edit this file, it is auto-generated ***

#include "psddl_hdf2psana/fli.ddl.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/Utils.h"
#include "PSEvt/DataProxy.h"
namespace psddl_hdf2psana {
namespace Fli {

hdf5pp::Type ns_ConfigV1_v0_dataset_config_stored_type()
{
  typedef ns_ConfigV1_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("width", offsetof(DsType, width), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("height", offsetof(DsType, height), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("orgX", offsetof(DsType, orgX), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("orgY", offsetof(DsType, orgY), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("binX", offsetof(DsType, binX), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("binY", offsetof(DsType, binY), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("exposureTime", offsetof(DsType, exposureTime), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("coolingTemp", offsetof(DsType, coolingTemp), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("gainIndex", offsetof(DsType, gainIndex), hdf5pp::TypeTraits<uint8_t>::stored_type());
  type.insert("readoutSpeedIndex", offsetof(DsType, readoutSpeedIndex), hdf5pp::TypeTraits<uint8_t>::stored_type());
  type.insert("exposureEventCode", offsetof(DsType, exposureEventCode), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("numDelayShots", offsetof(DsType, numDelayShots), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("frameSize", offsetof(DsType, frameSize), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("numPixelsX", offsetof(DsType, numPixelsX), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("numPixelsY", offsetof(DsType, numPixelsY), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("numPixels", offsetof(DsType, numPixels), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_ConfigV1_v0::dataset_config::stored_type()
{
  static hdf5pp::Type type = ns_ConfigV1_v0_dataset_config_stored_type();
  return type;
}

hdf5pp::Type ns_ConfigV1_v0_dataset_config_native_type()
{
  typedef ns_ConfigV1_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("width", offsetof(DsType, width), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("height", offsetof(DsType, height), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("orgX", offsetof(DsType, orgX), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("orgY", offsetof(DsType, orgY), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("binX", offsetof(DsType, binX), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("binY", offsetof(DsType, binY), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("exposureTime", offsetof(DsType, exposureTime), hdf5pp::TypeTraits<float>::native_type());
  type.insert("coolingTemp", offsetof(DsType, coolingTemp), hdf5pp::TypeTraits<float>::native_type());
  type.insert("gainIndex", offsetof(DsType, gainIndex), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("readoutSpeedIndex", offsetof(DsType, readoutSpeedIndex), hdf5pp::TypeTraits<uint8_t>::native_type());
  type.insert("exposureEventCode", offsetof(DsType, exposureEventCode), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("numDelayShots", offsetof(DsType, numDelayShots), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("frameSize", offsetof(DsType, frameSize), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numPixelsX", offsetof(DsType, numPixelsX), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numPixelsY", offsetof(DsType, numPixelsY), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("numPixels", offsetof(DsType, numPixels), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

hdf5pp::Type ns_ConfigV1_v0::dataset_config::native_type()
{
  static hdf5pp::Type type = ns_ConfigV1_v0_dataset_config_native_type();
  return type;
}
ns_ConfigV1_v0::dataset_config::dataset_config()
{
}
ns_ConfigV1_v0::dataset_config::~dataset_config()
{
}
uint32_t ConfigV1_v0::width() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint32_t(m_ds_config->width);
}
uint32_t ConfigV1_v0::height() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint32_t(m_ds_config->height);
}
uint32_t ConfigV1_v0::orgX() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint32_t(m_ds_config->orgX);
}
uint32_t ConfigV1_v0::orgY() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint32_t(m_ds_config->orgY);
}
uint32_t ConfigV1_v0::binX() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint32_t(m_ds_config->binX);
}
uint32_t ConfigV1_v0::binY() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint32_t(m_ds_config->binY);
}
float ConfigV1_v0::exposureTime() const {
  if (not m_ds_config.get()) read_ds_config();
  return float(m_ds_config->exposureTime);
}
float ConfigV1_v0::coolingTemp() const {
  if (not m_ds_config.get()) read_ds_config();
  return float(m_ds_config->coolingTemp);
}
uint8_t ConfigV1_v0::gainIndex() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint8_t(m_ds_config->gainIndex);
}
uint8_t ConfigV1_v0::readoutSpeedIndex() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint8_t(m_ds_config->readoutSpeedIndex);
}
uint16_t ConfigV1_v0::exposureEventCode() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint16_t(m_ds_config->exposureEventCode);
}
uint32_t ConfigV1_v0::numDelayShots() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint32_t(m_ds_config->numDelayShots);
}
uint32_t ConfigV1_v0::frameSize() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint32_t(m_ds_config->frameSize);
}
uint32_t ConfigV1_v0::numPixelsX() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint32_t(m_ds_config->numPixelsX);
}
uint32_t ConfigV1_v0::numPixelsY() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint32_t(m_ds_config->numPixelsY);
}
uint32_t ConfigV1_v0::numPixels() const {
  if (not m_ds_config.get()) read_ds_config();
  return uint32_t(m_ds_config->numPixels);
}
void ConfigV1_v0::read_ds_config() const {
  m_ds_config = hdf5pp::Utils::readGroup<Fli::ns_ConfigV1_v0::dataset_config>(m_group, "config", m_idx);
}
boost::shared_ptr<PSEvt::Proxy<Psana::Fli::ConfigV1> > make_ConfigV1(int version, hdf5pp::Group group, hsize_t idx) {
  switch (version) {
  case 0:
    return boost::make_shared<PSEvt::DataProxy<Psana::Fli::ConfigV1> >(boost::make_shared<ConfigV1_v0>(group, idx));
  default:
    return boost::make_shared<PSEvt::DataProxy<Psana::Fli::ConfigV1> >(boost::shared_ptr<Psana::Fli::ConfigV1>());
  }
}

hdf5pp::Type ns_FrameV1_v0_dataset_frame_stored_type()
{
  typedef ns_FrameV1_v0::dataset_frame DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("shotIdStart", offsetof(DsType, shotIdStart), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("readoutTime", offsetof(DsType, readoutTime), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("temperature", offsetof(DsType, temperature), hdf5pp::TypeTraits<float>::stored_type());
  return type;
}

hdf5pp::Type ns_FrameV1_v0::dataset_frame::stored_type()
{
  static hdf5pp::Type type = ns_FrameV1_v0_dataset_frame_stored_type();
  return type;
}

hdf5pp::Type ns_FrameV1_v0_dataset_frame_native_type()
{
  typedef ns_FrameV1_v0::dataset_frame DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("shotIdStart", offsetof(DsType, shotIdStart), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("readoutTime", offsetof(DsType, readoutTime), hdf5pp::TypeTraits<float>::native_type());
  type.insert("temperature", offsetof(DsType, temperature), hdf5pp::TypeTraits<float>::native_type());
  return type;
}

hdf5pp::Type ns_FrameV1_v0::dataset_frame::native_type()
{
  static hdf5pp::Type type = ns_FrameV1_v0_dataset_frame_native_type();
  return type;
}
ns_FrameV1_v0::dataset_frame::dataset_frame()
{
}
ns_FrameV1_v0::dataset_frame::~dataset_frame()
{
}
template <typename Config>
uint32_t FrameV1_v0<Config>::shotIdStart() const {
  if (not m_ds_frame.get()) read_ds_frame();
  return uint32_t(m_ds_frame->shotIdStart);
}
template <typename Config>
float FrameV1_v0<Config>::readoutTime() const {
  if (not m_ds_frame.get()) read_ds_frame();
  return float(m_ds_frame->readoutTime);
}
template <typename Config>
float FrameV1_v0<Config>::temperature() const {
  if (not m_ds_frame.get()) read_ds_frame();
  return float(m_ds_frame->temperature);
}
template <typename Config>
ndarray<const uint16_t, 2> FrameV1_v0<Config>::data() const {
  if (m_ds_data.empty()) read_ds_data();
  return m_ds_data;
}
template <typename Config>
void FrameV1_v0<Config>::read_ds_frame() const {
  m_ds_frame = hdf5pp::Utils::readGroup<Fli::ns_FrameV1_v0::dataset_frame>(m_group, "frame", m_idx);
}
template <typename Config>
void FrameV1_v0<Config>::read_ds_data() const {
  m_ds_data = hdf5pp::Utils::readNdarray<uint16_t, 2>(m_group, "data", m_idx);
}
template class FrameV1_v0<Psana::Fli::ConfigV1>;
boost::shared_ptr<PSEvt::Proxy<Psana::Fli::FrameV1> > make_FrameV1(int version, hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Psana::Fli::ConfigV1>& cfg) {
  switch (version) {
  case 0:
    return boost::make_shared<PSEvt::DataProxy<Psana::Fli::FrameV1> >(boost::make_shared<FrameV1_v0<Psana::Fli::ConfigV1> >(group, idx, cfg));
  default:
    return boost::make_shared<PSEvt::DataProxy<Psana::Fli::FrameV1> >(boost::shared_ptr<Psana::Fli::FrameV1>());
  }
}
} // namespace Fli
} // namespace psddl_hdf2psana
