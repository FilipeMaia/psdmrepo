
// *** Do not edit this file, it is auto-generated ***

#include "psddl_pdsdata/usdusb.ddl.h"

namespace PsddlPds {
namespace UsdUsb {
ndarray<const int32_t, 1>
DataV1::encoder_count() const {
  unsigned shape[1]={Encoder_Inputs};                 ndarray<int32_t,1> res(shape);                 for (unsigned i=0; i!=Encoder_Inputs; ++i) res[i]=int(this->_count[i] << 8)/256;                 return res;
}
} // namespace UsdUsb
} // namespace PsddlPds
