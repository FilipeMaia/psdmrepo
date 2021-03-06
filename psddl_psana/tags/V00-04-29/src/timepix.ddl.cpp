
// *** Do not edit this file, it is auto-generated ***

#include <cstddef>
#include "psddl_psana/timepix.ddl.h"
#include <iostream>
namespace Psana {
namespace Timepix {

ConfigV1::~ConfigV1() {}

std::ostream& operator<<(std::ostream& str, Timepix::ConfigV1::ReadoutSpeed enval) {
  const char* val;
  switch (enval) {
  case Timepix::ConfigV1::ReadoutSpeed_Slow:
    val = "ReadoutSpeed_Slow";
    break;
  case Timepix::ConfigV1::ReadoutSpeed_Fast:
    val = "ReadoutSpeed_Fast";
    break;
  default:
    return str << "ReadoutSpeed(" << int(enval) << ")";
  }
  return str << val;
}
std::ostream& operator<<(std::ostream& str, Timepix::ConfigV1::TriggerMode enval) {
  const char* val;
  switch (enval) {
  case Timepix::ConfigV1::TriggerMode_ExtPos:
    val = "TriggerMode_ExtPos";
    break;
  case Timepix::ConfigV1::TriggerMode_ExtNeg:
    val = "TriggerMode_ExtNeg";
    break;
  case Timepix::ConfigV1::TriggerMode_Soft:
    val = "TriggerMode_Soft";
    break;
  default:
    return str << "TriggerMode(" << int(enval) << ")";
  }
  return str << val;
}

ConfigV2::~ConfigV2() {}

std::ostream& operator<<(std::ostream& str, Timepix::ConfigV2::ReadoutSpeed enval) {
  const char* val;
  switch (enval) {
  case Timepix::ConfigV2::ReadoutSpeed_Slow:
    val = "ReadoutSpeed_Slow";
    break;
  case Timepix::ConfigV2::ReadoutSpeed_Fast:
    val = "ReadoutSpeed_Fast";
    break;
  default:
    return str << "ReadoutSpeed(" << int(enval) << ")";
  }
  return str << val;
}
std::ostream& operator<<(std::ostream& str, Timepix::ConfigV2::TriggerMode enval) {
  const char* val;
  switch (enval) {
  case Timepix::ConfigV2::TriggerMode_ExtPos:
    val = "TriggerMode_ExtPos";
    break;
  case Timepix::ConfigV2::TriggerMode_ExtNeg:
    val = "TriggerMode_ExtNeg";
    break;
  case Timepix::ConfigV2::TriggerMode_Soft:
    val = "TriggerMode_Soft";
    break;
  default:
    return str << "TriggerMode(" << int(enval) << ")";
  }
  return str << val;
}

ConfigV3::~ConfigV3() {}

std::ostream& operator<<(std::ostream& str, Timepix::ConfigV3::ReadoutSpeed enval) {
  const char* val;
  switch (enval) {
  case Timepix::ConfigV3::ReadoutSpeed_Slow:
    val = "ReadoutSpeed_Slow";
    break;
  case Timepix::ConfigV3::ReadoutSpeed_Fast:
    val = "ReadoutSpeed_Fast";
    break;
  default:
    return str << "ReadoutSpeed(" << int(enval) << ")";
  }
  return str << val;
}
std::ostream& operator<<(std::ostream& str, Timepix::ConfigV3::TimepixMode enval) {
  const char* val;
  switch (enval) {
  case Timepix::ConfigV3::TimepixMode_Counting:
    val = "TimepixMode_Counting";
    break;
  case Timepix::ConfigV3::TimepixMode_TOT:
    val = "TimepixMode_TOT";
    break;
  default:
    return str << "TimepixMode(" << int(enval) << ")";
  }
  return str << val;
}

DataV1::~DataV1() {}


DataV2::~DataV2() {}

} // namespace Timepix
} // namespace Psana
