
// *** Do not edit this file, it is auto-generated ***

#include "psddl_pdsdata/cspad2x2.ddl.h"

#include <iostream>
namespace PsddlPds {
namespace CsPad2x2 {
uint32_t
ConfigV1::numAsicsRead() const {
  return 4;
}
uint32_t
ConfigV1::numAsicsStored() const {
  return __builtin_popcount(this->roiMask())*2;
}
uint32_t
ConfigV2::numAsicsRead() const {
  return 4;
}
uint32_t
ConfigV2::numAsicsStored() const {
  return __builtin_popcount(this->roiMask())*2;
}
float
ElementV1::common_mode(uint32_t section) const {
  return 0;
}
} // namespace CsPad2x2
} // namespace PsddlPds
