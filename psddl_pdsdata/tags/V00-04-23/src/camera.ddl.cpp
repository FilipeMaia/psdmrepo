
// *** Do not edit this file, it is auto-generated ***

#include "psddl_pdsdata/camera.ddl.h"

#include <iostream>
namespace PsddlPds {
namespace Camera {
std::ostream& operator<<(std::ostream& str, Camera::FrameFexConfigV1::Forwarding enval) {
  const char* val;
  switch (enval) {
  case Camera::FrameFexConfigV1::NoFrame:
    val = "NoFrame";
    break;
  case Camera::FrameFexConfigV1::FullFrame:
    val = "FullFrame";
    break;
  case Camera::FrameFexConfigV1::RegionOfInterest:
    val = "RegionOfInterest";
    break;
  default:
    return str << "Forwarding(" << int(enval) << ")";
  }
  return str << val;
}
std::ostream& operator<<(std::ostream& str, Camera::FrameFexConfigV1::Processing enval) {
  const char* val;
  switch (enval) {
  case Camera::FrameFexConfigV1::NoProcessing:
    val = "NoProcessing";
    break;
  case Camera::FrameFexConfigV1::GssFullFrame:
    val = "GssFullFrame";
    break;
  case Camera::FrameFexConfigV1::GssRegionOfInterest:
    val = "GssRegionOfInterest";
    break;
  case Camera::FrameFexConfigV1::GssThreshold:
    val = "GssThreshold";
    break;
  default:
    return str << "Processing(" << int(enval) << ")";
  }
  return str << val;
}
ndarray<const uint8_t, 2>
FrameV1::data8() const {
  if (this->depth() > 8) return ndarray<const uint8_t, 2>(); return make_ndarray(_int_pixel_data().data(), height(), width());
}
ndarray<const uint16_t, 2>
FrameV1::data16() const {
  if (this->depth() <= 8) return ndarray<const uint16_t, 2>(); return make_ndarray((const uint16_t*)_int_pixel_data().data(), height(), width());
}
TwoDGaussianV1::TwoDGaussianV1()
{
}
TwoDGaussianV1::TwoDGaussianV1(uint64_t arg__integral, double arg__xmean, double arg__ymean, double arg__major_axis_width, double arg__minor_axis_width, double arg__major_axis_tilt)
    : _integral(arg__integral), _xmean(arg__xmean), _ymean(arg__ymean), _major_axis_width(arg__major_axis_width), _minor_axis_width(arg__minor_axis_width), _major_axis_tilt(arg__major_axis_tilt)
{
}
} // namespace Camera
} // namespace PsddlPds