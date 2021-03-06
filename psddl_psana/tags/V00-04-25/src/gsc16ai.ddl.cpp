
// *** Do not edit this file, it is auto-generated ***

#include <cstddef>
#include "psddl_psana/gsc16ai.ddl.h"
#include <iostream>
namespace Psana {
namespace Gsc16ai {

ConfigV1::~ConfigV1() {}

std::ostream& operator<<(std::ostream& str, Gsc16ai::ConfigV1::InputMode enval) {
  const char* val;
  switch (enval) {
  case Gsc16ai::ConfigV1::InputMode_Differential:
    val = "InputMode_Differential";
    break;
  case Gsc16ai::ConfigV1::InputMode_Zero:
    val = "InputMode_Zero";
    break;
  case Gsc16ai::ConfigV1::InputMode_Vref:
    val = "InputMode_Vref";
    break;
  default:
    return str << "InputMode(" << int(enval) << ")";
  }
  return str << val;
}
std::ostream& operator<<(std::ostream& str, Gsc16ai::ConfigV1::VoltageRange enval) {
  const char* val;
  switch (enval) {
  case Gsc16ai::ConfigV1::VoltageRange_10V:
    val = "VoltageRange_10V";
    break;
  case Gsc16ai::ConfigV1::VoltageRange_5V:
    val = "VoltageRange_5V";
    break;
  case Gsc16ai::ConfigV1::VoltageRange_2_5V:
    val = "VoltageRange_2_5V";
    break;
  default:
    return str << "VoltageRange(" << int(enval) << ")";
  }
  return str << val;
}
std::ostream& operator<<(std::ostream& str, Gsc16ai::ConfigV1::TriggerMode enval) {
  const char* val;
  switch (enval) {
  case Gsc16ai::ConfigV1::TriggerMode_ExtPos:
    val = "TriggerMode_ExtPos";
    break;
  case Gsc16ai::ConfigV1::TriggerMode_ExtNeg:
    val = "TriggerMode_ExtNeg";
    break;
  case Gsc16ai::ConfigV1::TriggerMode_IntClk:
    val = "TriggerMode_IntClk";
    break;
  default:
    return str << "TriggerMode(" << int(enval) << ")";
  }
  return str << val;
}
std::ostream& operator<<(std::ostream& str, Gsc16ai::ConfigV1::DataFormat enval) {
  const char* val;
  switch (enval) {
  case Gsc16ai::ConfigV1::DataFormat_TwosComplement:
    val = "DataFormat_TwosComplement";
    break;
  case Gsc16ai::ConfigV1::DataFormat_OffsetBinary:
    val = "DataFormat_OffsetBinary";
    break;
  default:
    return str << "DataFormat(" << int(enval) << ")";
  }
  return str << val;
}

DataV1::~DataV1() {}

} // namespace Gsc16ai
} // namespace Psana
