
// *** Do not edit this file, it is auto-generated ***

#include "psddl_pdsdata/epics.ddl.h"

namespace PsddlPds {
namespace Epics {
std::vector<int>
dbr_ctrl_short::units_shape() const {
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(MAX_UNITS_SIZE);
  return shape;
}
std::vector<int>
dbr_ctrl_float::units_shape() const {
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(MAX_UNITS_SIZE);
  return shape;
}
std::vector<int>
dbr_ctrl_enum::strings_shape() const {
  std::vector<int> shape;
  shape.reserve(2);
  shape.push_back(MAX_ENUM_STATES);
  shape.push_back(MAX_ENUM_STRING_SIZE);
  return shape;
}
std::vector<int>
dbr_ctrl_char::units_shape() const {
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(MAX_UNITS_SIZE);
  return shape;
}
std::vector<int>
dbr_ctrl_long::units_shape() const {
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(MAX_UNITS_SIZE);
  return shape;
}
std::vector<int>
dbr_ctrl_double::units_shape() const {
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(MAX_UNITS_SIZE);
  return shape;
}
uint8_t
EpicsPvHeader::isCtrl() const {
  return _iDbrType >= DBR_CTRL_STRING and _iDbrType <= DBR_CTRL_DOUBLE;
}
uint8_t
EpicsPvHeader::isTime() const {
  return _iDbrType >= DBR_TIME_STRING and _iDbrType <= DBR_TIME_DOUBLE;
}
std::vector<int>
EpicsPvCtrlHeader::pvName_shape() const {
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(iMaxPvNameLength);
  return shape;
}
const char*
EpicsPvCtrlString::value(uint32_t i) const {
  return data(i);
}
std::vector<int>
EpicsPvCtrlString::data_shape() const {
  std::vector<int> shape;
  shape.reserve(2);
  shape.push_back(this->numElements());
  shape.push_back( MAX_STRING_SIZE);
  return shape;
}
int16_t
EpicsPvCtrlShort::value(uint32_t i) const {
  return data()[i];
}
float
EpicsPvCtrlFloat::value(uint32_t i) const {
  return data()[i];
}
uint16_t
EpicsPvCtrlEnum::value(uint32_t i) const {
  return data()[i];
}
uint8_t
EpicsPvCtrlChar::value(uint32_t i) const {
  return data()[i];
}
int32_t
EpicsPvCtrlLong::value(uint32_t i) const {
  return data()[i];
}
double
EpicsPvCtrlDouble::value(uint32_t i) const {
  return data()[i];
}
const char*
EpicsPvTimeString::value(uint32_t i) const {
  return data(i);
}
std::vector<int>
EpicsPvTimeString::data_shape() const {
  std::vector<int> shape;
  shape.reserve(2);
  shape.push_back(this->numElements());
  shape.push_back( MAX_STRING_SIZE);
  return shape;
}
int16_t
EpicsPvTimeShort::value(uint32_t i) const {
  return data()[i];
}
float
EpicsPvTimeFloat::value(uint32_t i) const {
  return data()[i];
}
uint16_t
EpicsPvTimeEnum::value(uint32_t i) const {
  return data()[i];
}
uint8_t
EpicsPvTimeChar::value(uint32_t i) const {
  return data()[i];
}
int32_t
EpicsPvTimeLong::value(uint32_t i) const {
  return data()[i];
}
double
EpicsPvTimeDouble::value(uint32_t i) const {
  return data()[i];
}
std::vector<int>
ConfigV1::pvControls_shape() const {
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(this->_iNumPv);
  return shape;
}
} // namespace Epics
} // namespace PsddlPds
