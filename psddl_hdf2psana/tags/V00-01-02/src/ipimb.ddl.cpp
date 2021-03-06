
// *** Do not edit this file, it is auto-generated ***

#include "psddl_hdf2psana/ipimb.ddl.h"
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/VlenType.h"
#include "hdf5pp/Utils.h"
#include "PSEvt/DataProxy.h"
namespace psddl_hdf2psana {
namespace Ipimb {

hdf5pp::Type ns_ConfigV1_v0_dataset_config_stored_type()
{
  typedef ns_ConfigV1_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("triggerCounter", offsetof(DsType, triggerCounter), hdf5pp::TypeTraits<uint64_t>::stored_type());
  type.insert("serialID", offsetof(DsType, serialID), hdf5pp::TypeTraits<uint64_t>::stored_type());
  type.insert("chargeAmpRange", offsetof(DsType, chargeAmpRange), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("calibrationRange", offsetof(DsType, calibrationRange), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("resetLength", offsetof(DsType, resetLength), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("resetDelay", offsetof(DsType, resetDelay), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("chargeAmpRefVoltage", offsetof(DsType, chargeAmpRefVoltage), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("calibrationVoltage", offsetof(DsType, calibrationVoltage), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("diodeBias", offsetof(DsType, diodeBias), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("status", offsetof(DsType, status), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("errors", offsetof(DsType, errors), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("calStrobeLength", offsetof(DsType, calStrobeLength), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("trigDelay", offsetof(DsType, trigDelay), hdf5pp::TypeTraits<uint32_t>::stored_type());
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
  type.insert("triggerCounter", offsetof(DsType, triggerCounter), hdf5pp::TypeTraits<uint64_t>::native_type());
  type.insert("serialID", offsetof(DsType, serialID), hdf5pp::TypeTraits<uint64_t>::native_type());
  type.insert("chargeAmpRange", offsetof(DsType, chargeAmpRange), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("calibrationRange", offsetof(DsType, calibrationRange), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("resetLength", offsetof(DsType, resetLength), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("resetDelay", offsetof(DsType, resetDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("chargeAmpRefVoltage", offsetof(DsType, chargeAmpRefVoltage), hdf5pp::TypeTraits<float>::native_type());
  type.insert("calibrationVoltage", offsetof(DsType, calibrationVoltage), hdf5pp::TypeTraits<float>::native_type());
  type.insert("diodeBias", offsetof(DsType, diodeBias), hdf5pp::TypeTraits<float>::native_type());
  type.insert("status", offsetof(DsType, status), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("errors", offsetof(DsType, errors), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("calStrobeLength", offsetof(DsType, calStrobeLength), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("trigDelay", offsetof(DsType, trigDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
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
uint64_t ConfigV1_v0::triggerCounter() const {
  if (not m_ds_config) read_ds_config();
  return uint64_t(m_ds_config->triggerCounter);
}
uint64_t ConfigV1_v0::serialID() const {
  if (not m_ds_config) read_ds_config();
  return uint64_t(m_ds_config->serialID);
}
uint16_t ConfigV1_v0::chargeAmpRange() const {
  if (not m_ds_config) read_ds_config();
  return uint16_t(m_ds_config->chargeAmpRange);
}
uint16_t ConfigV1_v0::calibrationRange() const {
  if (not m_ds_config) read_ds_config();
  return uint16_t(m_ds_config->calibrationRange);
}
uint32_t ConfigV1_v0::resetLength() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->resetLength);
}
uint32_t ConfigV1_v0::resetDelay() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->resetDelay);
}
float ConfigV1_v0::chargeAmpRefVoltage() const {
  if (not m_ds_config) read_ds_config();
  return float(m_ds_config->chargeAmpRefVoltage);
}
float ConfigV1_v0::calibrationVoltage() const {
  if (not m_ds_config) read_ds_config();
  return float(m_ds_config->calibrationVoltage);
}
float ConfigV1_v0::diodeBias() const {
  if (not m_ds_config) read_ds_config();
  return float(m_ds_config->diodeBias);
}
uint16_t ConfigV1_v0::status() const {
  if (not m_ds_config) read_ds_config();
  return uint16_t(m_ds_config->status);
}
uint16_t ConfigV1_v0::errors() const {
  if (not m_ds_config) read_ds_config();
  return uint16_t(m_ds_config->errors);
}
uint16_t ConfigV1_v0::calStrobeLength() const {
  if (not m_ds_config) read_ds_config();
  return uint16_t(m_ds_config->calStrobeLength);
}
uint32_t ConfigV1_v0::trigDelay() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->trigDelay);
}

void ConfigV1_v0::read_ds_config() const {
  m_ds_config = hdf5pp::Utils::readGroup<Ipimb::ns_ConfigV1_v0::dataset_config>(m_group, "config", m_idx);
}
boost::shared_ptr<PSEvt::Proxy<Psana::Ipimb::ConfigV1> > make_ConfigV1(int version, hdf5pp::Group group, hsize_t idx) {
  switch (version) {
  case 0:
    return boost::make_shared<PSEvt::DataProxy<Psana::Ipimb::ConfigV1> >(boost::make_shared<ConfigV1_v0>(group, idx));
  default:
    return boost::make_shared<PSEvt::DataProxy<Psana::Ipimb::ConfigV1> >(boost::shared_ptr<Psana::Ipimb::ConfigV1>());
  }
}

hdf5pp::Type ns_ConfigV2_v0_dataset_config_stored_type()
{
  typedef ns_ConfigV2_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("triggerCounter", offsetof(DsType, triggerCounter), hdf5pp::TypeTraits<uint64_t>::stored_type());
  type.insert("serialID", offsetof(DsType, serialID), hdf5pp::TypeTraits<uint64_t>::stored_type());
  type.insert("chargeAmpRange", offsetof(DsType, chargeAmpRange), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("calibrationRange", offsetof(DsType, calibrationRange), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("resetLength", offsetof(DsType, resetLength), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("resetDelay", offsetof(DsType, resetDelay), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("chargeAmpRefVoltage", offsetof(DsType, chargeAmpRefVoltage), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("calibrationVoltage", offsetof(DsType, calibrationVoltage), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("diodeBias", offsetof(DsType, diodeBias), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("status", offsetof(DsType, status), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("errors", offsetof(DsType, errors), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("calStrobeLength", offsetof(DsType, calStrobeLength), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("trigDelay", offsetof(DsType, trigDelay), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("trigPsDelay", offsetof(DsType, trigPsDelay), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("adcDelay", offsetof(DsType, adcDelay), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_ConfigV2_v0::dataset_config::stored_type()
{
  static hdf5pp::Type type = ns_ConfigV2_v0_dataset_config_stored_type();
  return type;
}

hdf5pp::Type ns_ConfigV2_v0_dataset_config_native_type()
{
  typedef ns_ConfigV2_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("triggerCounter", offsetof(DsType, triggerCounter), hdf5pp::TypeTraits<uint64_t>::native_type());
  type.insert("serialID", offsetof(DsType, serialID), hdf5pp::TypeTraits<uint64_t>::native_type());
  type.insert("chargeAmpRange", offsetof(DsType, chargeAmpRange), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("calibrationRange", offsetof(DsType, calibrationRange), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("resetLength", offsetof(DsType, resetLength), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("resetDelay", offsetof(DsType, resetDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("chargeAmpRefVoltage", offsetof(DsType, chargeAmpRefVoltage), hdf5pp::TypeTraits<float>::native_type());
  type.insert("calibrationVoltage", offsetof(DsType, calibrationVoltage), hdf5pp::TypeTraits<float>::native_type());
  type.insert("diodeBias", offsetof(DsType, diodeBias), hdf5pp::TypeTraits<float>::native_type());
  type.insert("status", offsetof(DsType, status), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("errors", offsetof(DsType, errors), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("calStrobeLength", offsetof(DsType, calStrobeLength), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("trigDelay", offsetof(DsType, trigDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("trigPsDelay", offsetof(DsType, trigPsDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("adcDelay", offsetof(DsType, adcDelay), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

hdf5pp::Type ns_ConfigV2_v0::dataset_config::native_type()
{
  static hdf5pp::Type type = ns_ConfigV2_v0_dataset_config_native_type();
  return type;
}
ns_ConfigV2_v0::dataset_config::dataset_config()
{
}
ns_ConfigV2_v0::dataset_config::~dataset_config()
{
}
uint64_t ConfigV2_v0::triggerCounter() const {
  if (not m_ds_config) read_ds_config();
  return uint64_t(m_ds_config->triggerCounter);
}
uint64_t ConfigV2_v0::serialID() const {
  if (not m_ds_config) read_ds_config();
  return uint64_t(m_ds_config->serialID);
}
uint16_t ConfigV2_v0::chargeAmpRange() const {
  if (not m_ds_config) read_ds_config();
  return uint16_t(m_ds_config->chargeAmpRange);
}
uint16_t ConfigV2_v0::calibrationRange() const {
  if (not m_ds_config) read_ds_config();
  return uint16_t(m_ds_config->calibrationRange);
}
uint32_t ConfigV2_v0::resetLength() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->resetLength);
}
uint32_t ConfigV2_v0::resetDelay() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->resetDelay);
}
float ConfigV2_v0::chargeAmpRefVoltage() const {
  if (not m_ds_config) read_ds_config();
  return float(m_ds_config->chargeAmpRefVoltage);
}
float ConfigV2_v0::calibrationVoltage() const {
  if (not m_ds_config) read_ds_config();
  return float(m_ds_config->calibrationVoltage);
}
float ConfigV2_v0::diodeBias() const {
  if (not m_ds_config) read_ds_config();
  return float(m_ds_config->diodeBias);
}
uint16_t ConfigV2_v0::status() const {
  if (not m_ds_config) read_ds_config();
  return uint16_t(m_ds_config->status);
}
uint16_t ConfigV2_v0::errors() const {
  if (not m_ds_config) read_ds_config();
  return uint16_t(m_ds_config->errors);
}
uint16_t ConfigV2_v0::calStrobeLength() const {
  if (not m_ds_config) read_ds_config();
  return uint16_t(m_ds_config->calStrobeLength);
}
uint32_t ConfigV2_v0::trigDelay() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->trigDelay);
}
uint32_t ConfigV2_v0::trigPsDelay() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->trigPsDelay);
}
uint32_t ConfigV2_v0::adcDelay() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->adcDelay);
}

void ConfigV2_v0::read_ds_config() const {
  m_ds_config = hdf5pp::Utils::readGroup<Ipimb::ns_ConfigV2_v0::dataset_config>(m_group, "config", m_idx);
}
boost::shared_ptr<PSEvt::Proxy<Psana::Ipimb::ConfigV2> > make_ConfigV2(int version, hdf5pp::Group group, hsize_t idx) {
  switch (version) {
  case 0:
    return boost::make_shared<PSEvt::DataProxy<Psana::Ipimb::ConfigV2> >(boost::make_shared<ConfigV2_v0>(group, idx));
  default:
    return boost::make_shared<PSEvt::DataProxy<Psana::Ipimb::ConfigV2> >(boost::shared_ptr<Psana::Ipimb::ConfigV2>());
  }
}

hdf5pp::Type ns_DataV1_v0_dataset_data_stored_type()
{
  typedef ns_DataV1_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("triggerCounter", offsetof(DsType, triggerCounter), hdf5pp::TypeTraits<uint64_t>::stored_type());
  type.insert("config0", offsetof(DsType, config0), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("config1", offsetof(DsType, config1), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("config2", offsetof(DsType, config2), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel0", offsetof(DsType, channel0), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel1", offsetof(DsType, channel1), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel2", offsetof(DsType, channel2), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel3", offsetof(DsType, channel3), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("checksum", offsetof(DsType, checksum), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel0Volts", offsetof(DsType, channel0Volts), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("channel1Volts", offsetof(DsType, channel1Volts), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("channel2Volts", offsetof(DsType, channel2Volts), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("channel3Volts", offsetof(DsType, channel3Volts), hdf5pp::TypeTraits<float>::stored_type());
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
  type.insert("triggerCounter", offsetof(DsType, triggerCounter), hdf5pp::TypeTraits<uint64_t>::native_type());
  type.insert("config0", offsetof(DsType, config0), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("config1", offsetof(DsType, config1), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("config2", offsetof(DsType, config2), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel0", offsetof(DsType, channel0), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel1", offsetof(DsType, channel1), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel2", offsetof(DsType, channel2), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel3", offsetof(DsType, channel3), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("checksum", offsetof(DsType, checksum), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel0Volts", offsetof(DsType, channel0Volts), hdf5pp::TypeTraits<float>::native_type());
  type.insert("channel1Volts", offsetof(DsType, channel1Volts), hdf5pp::TypeTraits<float>::native_type());
  type.insert("channel2Volts", offsetof(DsType, channel2Volts), hdf5pp::TypeTraits<float>::native_type());
  type.insert("channel3Volts", offsetof(DsType, channel3Volts), hdf5pp::TypeTraits<float>::native_type());
  return type;
}

hdf5pp::Type ns_DataV1_v0::dataset_data::native_type()
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
uint64_t DataV1_v0::triggerCounter() const {
  if (not m_ds_data) read_ds_data();
  return uint64_t(m_ds_data->triggerCounter);
}
uint16_t DataV1_v0::config0() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->config0);
}
uint16_t DataV1_v0::config1() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->config1);
}
uint16_t DataV1_v0::config2() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->config2);
}
uint16_t DataV1_v0::channel0() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel0);
}
uint16_t DataV1_v0::channel1() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel1);
}
uint16_t DataV1_v0::channel2() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel2);
}
uint16_t DataV1_v0::channel3() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel3);
}
uint16_t DataV1_v0::checksum() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->checksum);
}
float DataV1_v0::channel0Volts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel0Volts);
}
float DataV1_v0::channel1Volts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel1Volts);
}
float DataV1_v0::channel2Volts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel2Volts);
}
float DataV1_v0::channel3Volts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel3Volts);
}
void DataV1_v0::read_ds_data() const {
  m_ds_data = hdf5pp::Utils::readGroup<Ipimb::ns_DataV1_v0::dataset_data>(m_group, "data", m_idx);
}
boost::shared_ptr<PSEvt::Proxy<Psana::Ipimb::DataV1> > make_DataV1(int version, hdf5pp::Group group, hsize_t idx) {
  switch (version) {
  case 0:
    return boost::make_shared<PSEvt::DataProxy<Psana::Ipimb::DataV1> >(boost::make_shared<DataV1_v0>(group, idx));
  default:
    return boost::make_shared<PSEvt::DataProxy<Psana::Ipimb::DataV1> >(boost::shared_ptr<Psana::Ipimb::DataV1>());
  }
}

hdf5pp::Type ns_DataV2_v0_dataset_data_stored_type()
{
  typedef ns_DataV2_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("config0", offsetof(DsType, config0), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("config1", offsetof(DsType, config1), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("config2", offsetof(DsType, config2), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel0", offsetof(DsType, channel0), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel1", offsetof(DsType, channel1), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel2", offsetof(DsType, channel2), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel3", offsetof(DsType, channel3), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel0ps", offsetof(DsType, channel0ps), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel1ps", offsetof(DsType, channel1ps), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel2ps", offsetof(DsType, channel2ps), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel3ps", offsetof(DsType, channel3ps), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("checksum", offsetof(DsType, checksum), hdf5pp::TypeTraits<uint16_t>::stored_type());
  type.insert("channel0Volts", offsetof(DsType, channel0Volts), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("channel1Volts", offsetof(DsType, channel1Volts), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("channel2Volts", offsetof(DsType, channel2Volts), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("channel3Volts", offsetof(DsType, channel3Volts), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("channel0psVolts", offsetof(DsType, channel0psVolts), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("channel1psVolts", offsetof(DsType, channel1psVolts), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("channel2psVolts", offsetof(DsType, channel2psVolts), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("channel3psVolts", offsetof(DsType, channel3psVolts), hdf5pp::TypeTraits<float>::stored_type());
  type.insert("triggerCounter", offsetof(DsType, triggerCounter), hdf5pp::TypeTraits<uint64_t>::stored_type());
  return type;
}

hdf5pp::Type ns_DataV2_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_DataV2_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_DataV2_v0_dataset_data_native_type()
{
  typedef ns_DataV2_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("config0", offsetof(DsType, config0), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("config1", offsetof(DsType, config1), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("config2", offsetof(DsType, config2), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel0", offsetof(DsType, channel0), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel1", offsetof(DsType, channel1), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel2", offsetof(DsType, channel2), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel3", offsetof(DsType, channel3), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel0ps", offsetof(DsType, channel0ps), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel1ps", offsetof(DsType, channel1ps), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel2ps", offsetof(DsType, channel2ps), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel3ps", offsetof(DsType, channel3ps), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("checksum", offsetof(DsType, checksum), hdf5pp::TypeTraits<uint16_t>::native_type());
  type.insert("channel0Volts", offsetof(DsType, channel0Volts), hdf5pp::TypeTraits<float>::native_type());
  type.insert("channel1Volts", offsetof(DsType, channel1Volts), hdf5pp::TypeTraits<float>::native_type());
  type.insert("channel2Volts", offsetof(DsType, channel2Volts), hdf5pp::TypeTraits<float>::native_type());
  type.insert("channel3Volts", offsetof(DsType, channel3Volts), hdf5pp::TypeTraits<float>::native_type());
  type.insert("channel0psVolts", offsetof(DsType, channel0psVolts), hdf5pp::TypeTraits<float>::native_type());
  type.insert("channel1psVolts", offsetof(DsType, channel1psVolts), hdf5pp::TypeTraits<float>::native_type());
  type.insert("channel2psVolts", offsetof(DsType, channel2psVolts), hdf5pp::TypeTraits<float>::native_type());
  type.insert("channel3psVolts", offsetof(DsType, channel3psVolts), hdf5pp::TypeTraits<float>::native_type());
  type.insert("triggerCounter", offsetof(DsType, triggerCounter), hdf5pp::TypeTraits<uint64_t>::native_type());
  return type;
}

hdf5pp::Type ns_DataV2_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_DataV2_v0_dataset_data_native_type();
  return type;
}
ns_DataV2_v0::dataset_data::dataset_data()
{
}
ns_DataV2_v0::dataset_data::~dataset_data()
{
}
uint16_t DataV2_v0::config0() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->config0);
}
uint16_t DataV2_v0::config1() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->config1);
}
uint16_t DataV2_v0::config2() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->config2);
}
uint16_t DataV2_v0::channel0() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel0);
}
uint16_t DataV2_v0::channel1() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel1);
}
uint16_t DataV2_v0::channel2() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel2);
}
uint16_t DataV2_v0::channel3() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel3);
}
uint16_t DataV2_v0::channel0ps() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel0ps);
}
uint16_t DataV2_v0::channel1ps() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel1ps);
}
uint16_t DataV2_v0::channel2ps() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel2ps);
}
uint16_t DataV2_v0::channel3ps() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->channel3ps);
}
uint16_t DataV2_v0::checksum() const {
  if (not m_ds_data) read_ds_data();
  return uint16_t(m_ds_data->checksum);
}
float DataV2_v0::channel0Volts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel0Volts);
}
float DataV2_v0::channel1Volts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel1Volts);
}
float DataV2_v0::channel2Volts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel2Volts);
}
float DataV2_v0::channel3Volts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel3Volts);
}
float DataV2_v0::channel0psVolts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel0psVolts);
}
float DataV2_v0::channel1psVolts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel1psVolts);
}
float DataV2_v0::channel2psVolts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel2psVolts);
}
float DataV2_v0::channel3psVolts() const {
  if (not m_ds_data) read_ds_data();
  return float(m_ds_data->channel3psVolts);
}
uint64_t DataV2_v0::triggerCounter() const {
  if (not m_ds_data) read_ds_data();
  return uint64_t(m_ds_data->triggerCounter);
}
void DataV2_v0::read_ds_data() const {
  m_ds_data = hdf5pp::Utils::readGroup<Ipimb::ns_DataV2_v0::dataset_data>(m_group, "data", m_idx);
}
boost::shared_ptr<PSEvt::Proxy<Psana::Ipimb::DataV2> > make_DataV2(int version, hdf5pp::Group group, hsize_t idx) {
  switch (version) {
  case 0:
    return boost::make_shared<PSEvt::DataProxy<Psana::Ipimb::DataV2> >(boost::make_shared<DataV2_v0>(group, idx));
  default:
    return boost::make_shared<PSEvt::DataProxy<Psana::Ipimb::DataV2> >(boost::shared_ptr<Psana::Ipimb::DataV2>());
  }
}
} // namespace Ipimb
} // namespace psddl_hdf2psana
