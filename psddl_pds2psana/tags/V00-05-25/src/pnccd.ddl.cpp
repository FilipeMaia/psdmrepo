
// *** Do not edit this file, it is auto-generated ***

#include "psddl_pds2psana/pnccd.ddl.h"

#include <cstddef>

#include <stdexcept>

namespace psddl_pds2psana {
namespace PNCCD {
ConfigV1::ConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::PNCCD::ConfigV1()
  , m_xtcObj(xtcPtr)
{
}
ConfigV1::~ConfigV1()
{
}


uint32_t ConfigV1::numLinks() const { return m_xtcObj->numLinks(); }

uint32_t ConfigV1::payloadSizePerLink() const { return m_xtcObj->payloadSizePerLink(); }
ConfigV2::ConfigV2(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::PNCCD::ConfigV2()
  , m_xtcObj(xtcPtr)
{
}
ConfigV2::~ConfigV2()
{
}


uint32_t ConfigV2::numLinks() const { return m_xtcObj->numLinks(); }

uint32_t ConfigV2::payloadSizePerLink() const { return m_xtcObj->payloadSizePerLink(); }

uint32_t ConfigV2::numChannels() const { return m_xtcObj->numChannels(); }

uint32_t ConfigV2::numRows() const { return m_xtcObj->numRows(); }

uint32_t ConfigV2::numSubmoduleChannels() const { return m_xtcObj->numSubmoduleChannels(); }

uint32_t ConfigV2::numSubmoduleRows() const { return m_xtcObj->numSubmoduleRows(); }

uint32_t ConfigV2::numSubmodules() const { return m_xtcObj->numSubmodules(); }

uint32_t ConfigV2::camexMagic() const { return m_xtcObj->camexMagic(); }

const char* ConfigV2::info() const { return m_xtcObj->info(); }

const char* ConfigV2::timingFName() const { return m_xtcObj->timingFName(); }

std::vector<int> ConfigV2::info_shape() const { return m_xtcObj->info_shape(); }

std::vector<int> ConfigV2::timingFName_shape() const { return m_xtcObj->timingFName_shape(); }
FrameV1::FrameV1(const boost::shared_ptr<const XtcType>& xtcPtr, const boost::shared_ptr<const PsddlPds::PNCCD::ConfigV1>& cfgPtr)
  : Psana::PNCCD::FrameV1()
  , m_xtcObj(xtcPtr)
  , m_cfgPtr0(cfgPtr)
{
}
FrameV1::FrameV1(const boost::shared_ptr<const XtcType>& xtcPtr, const boost::shared_ptr<const PsddlPds::PNCCD::ConfigV2>& cfgPtr)
  : Psana::PNCCD::FrameV1()
  , m_xtcObj(xtcPtr)
  , m_cfgPtr1(cfgPtr)
{
}
FrameV1::~FrameV1()
{
}


uint32_t FrameV1::specialWord() const { return m_xtcObj->specialWord(); }

uint32_t FrameV1::frameNumber() const { return m_xtcObj->frameNumber(); }

uint32_t FrameV1::timeStampHi() const { return m_xtcObj->timeStampHi(); }

uint32_t FrameV1::timeStampLo() const { return m_xtcObj->timeStampLo(); }

ndarray<const uint16_t, 1> FrameV1::_data() const {
  if (m_cfgPtr0.get()) return m_xtcObj->_data(*m_cfgPtr0);
  if (m_cfgPtr1.get()) return m_xtcObj->_data(*m_cfgPtr1);
  throw std::runtime_error("FrameV1::_data: config object pointer is zero");
}


ndarray<const uint16_t, 2> FrameV1::data() const {
  if (m_cfgPtr0.get()) return m_xtcObj->data(*m_cfgPtr0);
  if (m_cfgPtr1.get()) return m_xtcObj->data(*m_cfgPtr1);
  throw std::runtime_error("FrameV1::data: config object pointer is zero");
}

FullFrameV1::FullFrameV1(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::PNCCD::FullFrameV1()
  , m_xtcObj(xtcPtr)
{
}
FullFrameV1::~FullFrameV1()
{
}


uint32_t FullFrameV1::specialWord() const { return m_xtcObj->specialWord(); }

uint32_t FullFrameV1::frameNumber() const { return m_xtcObj->frameNumber(); }

uint32_t FullFrameV1::timeStampHi() const { return m_xtcObj->timeStampHi(); }

uint32_t FullFrameV1::timeStampLo() const { return m_xtcObj->timeStampLo(); }

ndarray<const uint16_t, 2> FullFrameV1::data() const { return m_xtcObj->data(); }
FramesV1::FramesV1(const boost::shared_ptr<const XtcType>& xtcPtr, const boost::shared_ptr<const PsddlPds::PNCCD::ConfigV1>& cfgPtr)
  : Psana::PNCCD::FramesV1()
  , m_xtcObj(xtcPtr)
  , m_cfgPtr0(cfgPtr)
{
  {
    const std::vector<int>& dims = xtcPtr->frame_shape(*cfgPtr);
    _frames.reserve(dims[0]);
    for (int i0=0; i0 != dims[0]; ++i0) {
      const PsddlPds::PNCCD::FrameV1& d = xtcPtr->frame(*cfgPtr, i0);
      boost::shared_ptr<const PsddlPds::PNCCD::FrameV1> dPtr(m_xtcObj, &d);
      _frames.push_back(psddl_pds2psana::PNCCD::FrameV1(dPtr, cfgPtr));
    }
  }
}
FramesV1::FramesV1(const boost::shared_ptr<const XtcType>& xtcPtr, const boost::shared_ptr<const PsddlPds::PNCCD::ConfigV2>& cfgPtr)
  : Psana::PNCCD::FramesV1()
  , m_xtcObj(xtcPtr)
  , m_cfgPtr1(cfgPtr)
{
  {
    const std::vector<int>& dims = xtcPtr->frame_shape(*cfgPtr);
    _frames.reserve(dims[0]);
    for (int i0=0; i0 != dims[0]; ++i0) {
      const PsddlPds::PNCCD::FrameV1& d = xtcPtr->frame(*cfgPtr, i0);
      boost::shared_ptr<const PsddlPds::PNCCD::FrameV1> dPtr(m_xtcObj, &d);
      _frames.push_back(psddl_pds2psana::PNCCD::FrameV1(dPtr, cfgPtr));
    }
  }
}
FramesV1::~FramesV1()
{
}


const Psana::PNCCD::FrameV1& FramesV1::frame(uint32_t i0) const { return _frames[i0]; }

uint32_t FramesV1::numLinks() const {
  if (m_cfgPtr0.get()) return m_xtcObj->numLinks(*m_cfgPtr0);
  if (m_cfgPtr1.get()) return m_xtcObj->numLinks(*m_cfgPtr1);
  throw std::runtime_error("FramesV1::numLinks: config object pointer is zero");
}

std::vector<int> FramesV1::frame_shape() const
{
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(_frames.size());
  return shape;
}

} // namespace PNCCD
} // namespace psddl_pds2psana