
// *** Do not edit this file, it is auto-generated ***

#include "psddl_pdsdata/ipimb.ddl.h"

namespace PsddlPds {
namespace Ipimb {
float
DataV1::channel0Volts() const {
  return float(this->_channel0)*3.3/65535;
}
float
DataV1::channel1Volts() const {
  return float(this->_channel1)*3.3/65535;
}
float
DataV1::channel2Volts() const {
  return float(this->_channel2)*3.3/65535;
}
float
DataV1::channel3Volts() const {
  return float(this->_channel3)*3.3/65535;
}
float
DataV2::channel0Volts() const {
  return float(this->_channel0)*ipimbAdcRange/(ipimbAdcSteps - 1);
}
float
DataV2::channel1Volts() const {
  return float(this->_channel1)*ipimbAdcRange/(ipimbAdcSteps - 1);
}
float
DataV2::channel2Volts() const {
  return float(this->_channel2)*ipimbAdcRange/(ipimbAdcSteps - 1);
}
float
DataV2::channel3Volts() const {
  return float(this->_channel3)*ipimbAdcRange/(ipimbAdcSteps - 1);
}
float
DataV2::channel0psVolts() const {
  return float(this->_channel0ps)*ipimbAdcRange/(ipimbAdcSteps - 1);
}
float
DataV2::channel1psVolts() const {
  return float(this->_channel1ps)*ipimbAdcRange/(ipimbAdcSteps - 1);
}
float
DataV2::channel2psVolts() const {
  return float(this->_channel2ps)*ipimbAdcRange/(ipimbAdcSteps - 1);
}
float
DataV2::channel3psVolts() const {
  return float(this->_channel3ps)*ipimbAdcRange/(ipimbAdcSteps - 1);
}
} // namespace Ipimb
} // namespace PsddlPds
