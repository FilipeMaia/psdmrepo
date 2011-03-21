
// *** Do not edit this file, it is auto-generated ***

#include "psddl_pds2psana/acqiris.ddl.h"

#include <cstddef>

#include <stdexcept>

namespace psddl_pds2psana {
namespace Acqiris {
Psana::Acqiris::VertV1::Coupling pds_to_psana(PsddlPds::Acqiris::VertV1::Coupling e)
{
  return Psana::Acqiris::VertV1::Coupling(e);
}

Psana::Acqiris::VertV1::Bandwidth pds_to_psana(PsddlPds::Acqiris::VertV1::Bandwidth e)
{
  return Psana::Acqiris::VertV1::Bandwidth(e);
}

Psana::Acqiris::VertV1 pds_to_psana(PsddlPds::Acqiris::VertV1 pds)
{
  return Psana::Acqiris::VertV1(pds.fullScale(), pds.offset(), pds.coupling(), pds.bandwidth());
}

Psana::Acqiris::HorizV1 pds_to_psana(PsddlPds::Acqiris::HorizV1 pds)
{
  return Psana::Acqiris::HorizV1(pds.sampInterval(), pds.delayTime(), pds.nbrSamples(), pds.nbrSegments());
}

Psana::Acqiris::TrigV1::Source pds_to_psana(PsddlPds::Acqiris::TrigV1::Source e)
{
  return Psana::Acqiris::TrigV1::Source(e);
}

Psana::Acqiris::TrigV1::Coupling pds_to_psana(PsddlPds::Acqiris::TrigV1::Coupling e)
{
  return Psana::Acqiris::TrigV1::Coupling(e);
}

Psana::Acqiris::TrigV1::Slope pds_to_psana(PsddlPds::Acqiris::TrigV1::Slope e)
{
  return Psana::Acqiris::TrigV1::Slope(e);
}

Psana::Acqiris::TrigV1 pds_to_psana(PsddlPds::Acqiris::TrigV1 pds)
{
  return Psana::Acqiris::TrigV1(pds.coupling(), pds.input(), pds.slope(), pds.level());
}

ConfigV1::ConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::Acqiris::ConfigV1()
  , m_xtcObj(xtcPtr)
  , _trig(psddl_pds2psana::Acqiris::pds_to_psana(xtcPtr->trig()))
  , _horiz(psddl_pds2psana::Acqiris::pds_to_psana(xtcPtr->horiz()))
{
  {
    const std::vector<int>& dims = xtcPtr->_vert_shape();
    _vert.reserve(dims[0]);
    for (int i0=0; i0 != dims[0]; ++i0) {
      _vert.push_back(psddl_pds2psana::Acqiris::pds_to_psana(xtcPtr->vert(i0)));
    }
  }
}
ConfigV1::~ConfigV1()
{
}


uint32_t ConfigV1::nbrConvertersPerChannel() const { return m_xtcObj->nbrConvertersPerChannel(); }

uint32_t ConfigV1::channelMask() const { return m_xtcObj->channelMask(); }

uint32_t ConfigV1::nbrBanks() const { return m_xtcObj->nbrBanks(); }

const Psana::Acqiris::TrigV1& ConfigV1::trig() const { return _trig; }

const Psana::Acqiris::HorizV1& ConfigV1::horiz() const { return _horiz; }

const Psana::Acqiris::VertV1& ConfigV1::vert(uint32_t i0) const { return _vert[i0]; }

uint32_t ConfigV1::nbrChannels() const { return m_xtcObj->nbrChannels(); }
std::vector<int> ConfigV1::_vert_shape() const
{
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(_vert.size());
  return shape;
}

Psana::Acqiris::TimestampV1 pds_to_psana(PsddlPds::Acqiris::TimestampV1 pds)
{
  return Psana::Acqiris::TimestampV1(pds.pos(), pds.timeStampLo(), pds.timeStampHi());
}

DataDescV1Elem::DataDescV1Elem(const boost::shared_ptr<const XtcType>& xtcPtr, const boost::shared_ptr<const PsddlPds::Acqiris::ConfigV1>& cfgPtr)
  : Psana::Acqiris::DataDescV1Elem()
  , m_xtcObj(xtcPtr)
  , m_cfgPtr0(cfgPtr)
{
  {
    const std::vector<int>& dims = xtcPtr->_timestamps_shape(*cfgPtr);
    _timestamps.reserve(dims[0]);
    for (int i0=0; i0 != dims[0]; ++i0) {
      _timestamps.push_back(psddl_pds2psana::Acqiris::pds_to_psana(xtcPtr->_timestamp(i0)));
    }
  }
}
DataDescV1Elem::~DataDescV1Elem()
{
}


uint32_t DataDescV1Elem::nbrSamplesInSeg() const { return m_xtcObj->nbrSamplesInSeg(); }

uint32_t DataDescV1Elem::indexFirstPoint() const { return m_xtcObj->indexFirstPoint(); }

uint32_t DataDescV1Elem::nbrSegments() const { return m_xtcObj->nbrSegments(); }

const Psana::Acqiris::TimestampV1& DataDescV1Elem::_timestamp(uint32_t i0) const { return _timestamps[i0]; }

const int16_t* DataDescV1Elem::_waveform() const {
  if (m_cfgPtr0.get()) return m_xtcObj->_waveform(*m_cfgPtr0);
  throw std::runtime_error("DataDescV1Elem::_waveform: config object pointer is zero");
}

std::vector<int> DataDescV1Elem::_timestamps_shape() const
{
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(_timestamps.size());
  return shape;
}


std::vector<int> DataDescV1Elem::_waveforms_shape() const {
  if (m_cfgPtr0.get()) return m_xtcObj->_waveforms_shape(*m_cfgPtr0);
  throw std::runtime_error("DataDescV1Elem::_waveforms_shape: config object pointer is zero");
}


std::vector<int> DataDescV1Elem::_extraSpace_shape() const { return m_xtcObj->_extraSpace_shape(); }
DataDescV1::DataDescV1(const boost::shared_ptr<const XtcType>& xtcPtr, const boost::shared_ptr<const PsddlPds::Acqiris::ConfigV1>& cfgPtr)
  : Psana::Acqiris::DataDescV1()
  , m_xtcObj(xtcPtr)
  , m_cfgPtr0(cfgPtr)
{
  {
    const std::vector<int>& dims = xtcPtr->_data_shape(*cfgPtr);
    _data.reserve(dims[0]);
    for (int i0=0; i0 != dims[0]; ++i0) {
      const PsddlPds::Acqiris::DataDescV1Elem& d = xtcPtr->data(i0);
      boost::shared_ptr<const PsddlPds::Acqiris::DataDescV1Elem> dPtr(m_xtcObj, &d);
      _data.push_back(psddl_pds2psana::Acqiris::DataDescV1Elem(dPtr, cfgPtr));
    }
  }
}
DataDescV1::~DataDescV1()
{
}


const Psana::Acqiris::DataDescV1Elem& DataDescV1::data(uint32_t i0) const { return _data[i0]; }
std::vector<int> DataDescV1::_data_shape() const
{
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(_data.size());
  return shape;
}

Psana::Acqiris::TdcChannel::Channel pds_to_psana(PsddlPds::Acqiris::TdcChannel::Channel e)
{
  return Psana::Acqiris::TdcChannel::Channel(e);
}

Psana::Acqiris::TdcChannel::Mode pds_to_psana(PsddlPds::Acqiris::TdcChannel::Mode e)
{
  return Psana::Acqiris::TdcChannel::Mode(e);
}

Psana::Acqiris::TdcChannel::Slope pds_to_psana(PsddlPds::Acqiris::TdcChannel::Slope e)
{
  return Psana::Acqiris::TdcChannel::Slope(e);
}

Psana::Acqiris::TdcChannel pds_to_psana(PsddlPds::Acqiris::TdcChannel pds)
{
  return Psana::Acqiris::TdcChannel(pds._channel_int(), pds._mode_int(), pds.level());
}

Psana::Acqiris::TdcAuxIO::Channel pds_to_psana(PsddlPds::Acqiris::TdcAuxIO::Channel e)
{
  return Psana::Acqiris::TdcAuxIO::Channel(e);
}

Psana::Acqiris::TdcAuxIO::Mode pds_to_psana(PsddlPds::Acqiris::TdcAuxIO::Mode e)
{
  return Psana::Acqiris::TdcAuxIO::Mode(e);
}

Psana::Acqiris::TdcAuxIO::Termination pds_to_psana(PsddlPds::Acqiris::TdcAuxIO::Termination e)
{
  return Psana::Acqiris::TdcAuxIO::Termination(e);
}

Psana::Acqiris::TdcAuxIO pds_to_psana(PsddlPds::Acqiris::TdcAuxIO pds)
{
  return Psana::Acqiris::TdcAuxIO(pds.channel_int(), pds.signal_int(), pds.qualifier_int());
}

Psana::Acqiris::TdcVetoIO::Channel pds_to_psana(PsddlPds::Acqiris::TdcVetoIO::Channel e)
{
  return Psana::Acqiris::TdcVetoIO::Channel(e);
}

Psana::Acqiris::TdcVetoIO::Mode pds_to_psana(PsddlPds::Acqiris::TdcVetoIO::Mode e)
{
  return Psana::Acqiris::TdcVetoIO::Mode(e);
}

Psana::Acqiris::TdcVetoIO::Termination pds_to_psana(PsddlPds::Acqiris::TdcVetoIO::Termination e)
{
  return Psana::Acqiris::TdcVetoIO::Termination(e);
}

Psana::Acqiris::TdcVetoIO pds_to_psana(PsddlPds::Acqiris::TdcVetoIO pds)
{
  return Psana::Acqiris::TdcVetoIO(pds.signal_int(), pds.qualifier_int());
}

TdcConfigV1::TdcConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::Acqiris::TdcConfigV1()
  , m_xtcObj(xtcPtr)
  , _veto(psddl_pds2psana::Acqiris::pds_to_psana(xtcPtr->veto()))
{
  {
    const std::vector<int>& dims = xtcPtr->_channel_shape();
    _channel.reserve(dims[0]);
    for (int i0=0; i0 != dims[0]; ++i0) {
      _channel.push_back(psddl_pds2psana::Acqiris::pds_to_psana(xtcPtr->channels(i0)));
    }
  }
  {
    const std::vector<int>& dims = xtcPtr->_auxIO_shape();
    _auxIO.reserve(dims[0]);
    for (int i0=0; i0 != dims[0]; ++i0) {
      _auxIO.push_back(psddl_pds2psana::Acqiris::pds_to_psana(xtcPtr->auxio(i0)));
    }
  }
}
TdcConfigV1::~TdcConfigV1()
{
}


const Psana::Acqiris::TdcChannel& TdcConfigV1::channels(uint32_t i0) const { return _channel[i0]; }

const Psana::Acqiris::TdcAuxIO& TdcConfigV1::auxio(uint32_t i0) const { return _auxIO[i0]; }

const Psana::Acqiris::TdcVetoIO& TdcConfigV1::veto() const { return _veto; }
std::vector<int> TdcConfigV1::_channel_shape() const
{
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(_channel.size());
  return shape;
}

std::vector<int> TdcConfigV1::_auxIO_shape() const
{
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(_auxIO.size());
  return shape;
}

Psana::Acqiris::TdcDataV1_Item::Source pds_to_psana(PsddlPds::Acqiris::TdcDataV1_Item::Source e)
{
  return Psana::Acqiris::TdcDataV1_Item::Source(e);
}

Psana::Acqiris::TdcDataV1_Item pds_to_psana(PsddlPds::Acqiris::TdcDataV1_Item pds)
{
  return Psana::Acqiris::TdcDataV1_Item(pds.value());
}

Psana::Acqiris::TdcDataV1Marker::Type pds_to_psana(PsddlPds::Acqiris::TdcDataV1Marker::Type e)
{
  return Psana::Acqiris::TdcDataV1Marker::Type(e);
}

TdcDataV1::TdcDataV1(const boost::shared_ptr<const XtcType>& xtcPtr)
  : Psana::Acqiris::TdcDataV1()
  , m_xtcObj(xtcPtr)
{
  {
    const std::vector<int>& dims = xtcPtr->_data_shape();
    _data.reserve(dims[0]);
    for (int i0=0; i0 != dims[0]; ++i0) {
      _data.push_back(psddl_pds2psana::Acqiris::pds_to_psana(xtcPtr->data(i0)));
    }
  }
}
TdcDataV1::~TdcDataV1()
{
}


const Psana::Acqiris::TdcDataV1_Item& TdcDataV1::data(uint32_t i0) const { return _data[i0]; }
std::vector<int> TdcDataV1::_data_shape() const
{
  std::vector<int> shape;
  shape.reserve(1);
  shape.push_back(_data.size());
  return shape;
}

} // namespace Acqiris
} // namespace psddl_pds2psana
