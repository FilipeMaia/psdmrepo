
// *** Do not edit this file, it is auto-generated ***

#include "psddl_pds2psana/usdusb.ddl.h"

#include <cstddef>

#include <stdexcept>

namespace psddl_pds2psana {
namespace UsdUsb {
Psana::UsdUsb::ConfigV1::Count_Mode pds_to_psana(PsddlPds::UsdUsb::ConfigV1::Count_Mode e)
{
  return Psana::UsdUsb::ConfigV1::Count_Mode(e);
}

Psana::UsdUsb::ConfigV1::Quad_Mode pds_to_psana(PsddlPds::UsdUsb::ConfigV1::Quad_Mode e)
{
  return Psana::UsdUsb::ConfigV1::Quad_Mode(e);
}

ConfigV1::ConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::UsdUsb::ConfigV1()
  , m_xtcObj(xtcPtr)
{
}
ConfigV1::~ConfigV1()
{
}


ndarray<uint32_t, 1> ConfigV1::counting_mode() const { return m_xtcObj->counting_mode(); }

ndarray<uint32_t, 1> ConfigV1::quadrature_mode() const { return m_xtcObj->quadrature_mode(); }
DataV1::DataV1(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::UsdUsb::DataV1()
  , m_xtcObj(xtcPtr)
{
}
DataV1::~DataV1()
{
}


uint8_t DataV1::digital_in() const { return m_xtcObj->digital_in(); }

uint32_t DataV1::timestamp() const { return m_xtcObj->timestamp(); }

int32_t DataV1::value(uint32_t i) const { return m_xtcObj->value(i); }
} // namespace UsdUsb
} // namespace psddl_pds2psana
