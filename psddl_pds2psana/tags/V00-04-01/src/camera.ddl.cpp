
// *** Do not edit this file, it is auto-generated ***

#include "psddl_pds2psana/camera.ddl.h"

#include <cstddef>

#include <stdexcept>

namespace psddl_pds2psana {
namespace Camera {
Psana::Camera::FrameCoord pds_to_psana(PsddlPds::Camera::FrameCoord pds)
{
  return Psana::Camera::FrameCoord(pds.column(), pds.row());
}

FrameFccdConfigV1::FrameFccdConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::Camera::FrameFccdConfigV1()
  , m_xtcObj(xtcPtr)
{
}
FrameFccdConfigV1::~FrameFccdConfigV1()
{
}

Psana::Camera::FrameFexConfigV1::Forwarding pds_to_psana(PsddlPds::Camera::FrameFexConfigV1::Forwarding e)
{
  return Psana::Camera::FrameFexConfigV1::Forwarding(e);
}

Psana::Camera::FrameFexConfigV1::Processing pds_to_psana(PsddlPds::Camera::FrameFexConfigV1::Processing e)
{
  return Psana::Camera::FrameFexConfigV1::Processing(e);
}

FrameFexConfigV1::FrameFexConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::Camera::FrameFexConfigV1()
  , m_xtcObj(xtcPtr)
  , _roiBegin(psddl_pds2psana::Camera::pds_to_psana(xtcPtr->roiBegin()))
  , _roiEnd(psddl_pds2psana::Camera::pds_to_psana(xtcPtr->roiEnd()))
{
  {
    typedef ndarray<PsddlPds::Camera::FrameCoord, 1> XtcNDArray;
    const XtcNDArray& xtc_ndarr = xtcPtr->masked_pixel_coordinates();
    _masked_pixel_coordinates_ndarray_storage_.reserve(xtc_ndarr.size());
    for (XtcNDArray::const_iterator it = xtc_ndarr.begin(); it != xtc_ndarr.end(); ++ it) {
      _masked_pixel_coordinates_ndarray_storage_.push_back(psddl_pds2psana::Camera::pds_to_psana(*it));
    }
    const unsigned* shape = xtc_ndarr.shape();
    std::copy(shape, shape+1, _masked_pixel_coordinates_ndarray_shape_);
  }
}
FrameFexConfigV1::~FrameFexConfigV1()
{
}


Psana::Camera::FrameFexConfigV1::Forwarding FrameFexConfigV1::forwarding() const { return pds_to_psana(m_xtcObj->forwarding()); }

uint32_t FrameFexConfigV1::forward_prescale() const { return m_xtcObj->forward_prescale(); }

Psana::Camera::FrameFexConfigV1::Processing FrameFexConfigV1::processing() const { return pds_to_psana(m_xtcObj->processing()); }

const Psana::Camera::FrameCoord& FrameFexConfigV1::roiBegin() const { return _roiBegin; }

const Psana::Camera::FrameCoord& FrameFexConfigV1::roiEnd() const { return _roiEnd; }

uint32_t FrameFexConfigV1::threshold() const { return m_xtcObj->threshold(); }

uint32_t FrameFexConfigV1::number_of_masked_pixels() const { return m_xtcObj->number_of_masked_pixels(); }

ndarray<Psana::Camera::FrameCoord, 1> FrameFexConfigV1::masked_pixel_coordinates() const { return ndarray<Psana::Camera::FrameCoord, 1>(&_masked_pixel_coordinates_ndarray_storage_[0], _masked_pixel_coordinates_ndarray_shape_); }
FrameV1::FrameV1(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::Camera::FrameV1()
  , m_xtcObj(xtcPtr)
{
}
FrameV1::~FrameV1()
{
}


uint32_t FrameV1::width() const { return m_xtcObj->width(); }

uint32_t FrameV1::height() const { return m_xtcObj->height(); }

uint32_t FrameV1::depth() const { return m_xtcObj->depth(); }

uint32_t FrameV1::offset() const { return m_xtcObj->offset(); }

ndarray<uint8_t, 1> FrameV1::_int_pixel_data() const { return m_xtcObj->_int_pixel_data(); }

ndarray<uint8_t, 2> FrameV1::data8() const { return m_xtcObj->data8(); }

ndarray<uint16_t, 2> FrameV1::data16() const { return m_xtcObj->data16(); }
TwoDGaussianV1::TwoDGaussianV1(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::Camera::TwoDGaussianV1()
  , m_xtcObj(xtcPtr)
{
}
TwoDGaussianV1::~TwoDGaussianV1()
{
}


uint64_t TwoDGaussianV1::integral() const { return m_xtcObj->integral(); }

double TwoDGaussianV1::xmean() const { return m_xtcObj->xmean(); }

double TwoDGaussianV1::ymean() const { return m_xtcObj->ymean(); }

double TwoDGaussianV1::major_axis_width() const { return m_xtcObj->major_axis_width(); }

double TwoDGaussianV1::minor_axis_width() const { return m_xtcObj->minor_axis_width(); }

double TwoDGaussianV1::major_axis_tilt() const { return m_xtcObj->major_axis_tilt(); }
} // namespace Camera
} // namespace psddl_pds2psana
