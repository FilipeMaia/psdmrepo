
// *** Do not edit this file, it is auto-generated ***

#include "psddl_pdsdata/acqiris.ddl.h"

namespace PsddlPds {
namespace Acqiris {
double
VertV1::slope() const {
  return _fullScale / ((1 << Acqiris::DataDescV1Elem::NumberOfBits)*(1 << Acqiris::DataDescV1Elem::BitShift));
}
uint32_t
ConfigV1::nbrChannels() const {
  return __builtin_popcount(this->_channelMask);
}
uint64_t
TimestampV1::value() const {
  return (((uint64_t)this->_timeStampHi)<<32) + this->_timeStampLo;
}
std::vector<int>
DataDescV1::data_shape(const Acqiris::ConfigV1& cfg) const {
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(cfg.nbrChannels());
  return shape;
}
uint32_t
TdcDataV1Common::nhits() const {
  return this->bf_val_();
}
uint8_t
TdcDataV1Common::overflow() const {
  return this->bf_ofv_();
}
uint32_t
TdcDataV1Channel::ticks() const {
  return this->bf_val_();
}
uint8_t
TdcDataV1Channel::overflow() const {
  return this->bf_ofv_();
}
double
TdcDataV1Channel::time() const {
  return this->bf_val_() * 50e-12;
}
Acqiris::TdcDataV1Marker::Type
TdcDataV1Marker::type() const {
  return Type(this->bf_val_());
}
} // namespace Acqiris
} // namespace PsddlPds
