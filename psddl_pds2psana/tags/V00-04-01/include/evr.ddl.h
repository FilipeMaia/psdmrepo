#ifndef PSDDL_PDS2PSANA_EVR_DDL_H
#define PSDDL_PDS2PSANA_EVR_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include <vector>
#include <boost/shared_ptr.hpp>
#include "psddl_psana/evr.ddl.h"
#include "psddl_pdsdata/evr.ddl.h"
#include "pdsdata/xtc/DetInfo.hh"
namespace psddl_pds2psana {
namespace EvrData {
Psana::EvrData::PulseConfig pds_to_psana(PsddlPds::EvrData::PulseConfig pds);

Psana::EvrData::PulseConfigV3 pds_to_psana(PsddlPds::EvrData::PulseConfigV3 pds);

Psana::EvrData::EventCodeV3 pds_to_psana(PsddlPds::EvrData::EventCodeV3 pds);

Psana::EvrData::EventCodeV4 pds_to_psana(PsddlPds::EvrData::EventCodeV4 pds);

Psana::EvrData::EventCodeV5 pds_to_psana(PsddlPds::EvrData::EventCodeV5 pds);

Psana::EvrData::OutputMap pds_to_psana(PsddlPds::EvrData::OutputMap pds);


class ConfigV1 : public Psana::EvrData::ConfigV1 {
public:
  typedef PsddlPds::EvrData::ConfigV1 XtcType;
  typedef Psana::EvrData::ConfigV1 PsanaType;
  ConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~ConfigV1();
  virtual uint32_t npulses() const;
  virtual uint32_t noutputs() const;
  virtual ndarray<Psana::EvrData::PulseConfig, 1> pulses() const;
  virtual ndarray<Psana::EvrData::OutputMap, 1> output_maps() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector<Psana::EvrData::PulseConfig> _pulses_ndarray_storage_;
  unsigned _pulses_ndarray_shape_[1];
  std::vector<Psana::EvrData::OutputMap> _output_maps_ndarray_storage_;
  unsigned _output_maps_ndarray_shape_[1];
};


class ConfigV2 : public Psana::EvrData::ConfigV2 {
public:
  typedef PsddlPds::EvrData::ConfigV2 XtcType;
  typedef Psana::EvrData::ConfigV2 PsanaType;
  ConfigV2(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~ConfigV2();
  virtual uint32_t opcode() const;
  virtual uint32_t npulses() const;
  virtual uint32_t noutputs() const;
  virtual ndarray<Psana::EvrData::PulseConfig, 1> pulses() const;
  virtual ndarray<Psana::EvrData::OutputMap, 1> output_maps() const;
  virtual Psana::EvrData::ConfigV2::BeamCode beam() const;
  virtual Psana::EvrData::ConfigV2::RateCode rate() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector<Psana::EvrData::PulseConfig> _pulses_ndarray_storage_;
  unsigned _pulses_ndarray_shape_[1];
  std::vector<Psana::EvrData::OutputMap> _output_maps_ndarray_storage_;
  unsigned _output_maps_ndarray_shape_[1];
};


class ConfigV3 : public Psana::EvrData::ConfigV3 {
public:
  typedef PsddlPds::EvrData::ConfigV3 XtcType;
  typedef Psana::EvrData::ConfigV3 PsanaType;
  ConfigV3(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~ConfigV3();
  virtual uint32_t neventcodes() const;
  virtual uint32_t npulses() const;
  virtual uint32_t noutputs() const;
  virtual ndarray<Psana::EvrData::EventCodeV3, 1> eventcodes() const;
  virtual ndarray<Psana::EvrData::PulseConfigV3, 1> pulses() const;
  virtual ndarray<Psana::EvrData::OutputMap, 1> output_maps() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector<Psana::EvrData::EventCodeV3> _eventcodes_ndarray_storage_;
  unsigned _eventcodes_ndarray_shape_[1];
  std::vector<Psana::EvrData::PulseConfigV3> _pulses_ndarray_storage_;
  unsigned _pulses_ndarray_shape_[1];
  std::vector<Psana::EvrData::OutputMap> _output_maps_ndarray_storage_;
  unsigned _output_maps_ndarray_shape_[1];
};


class ConfigV4 : public Psana::EvrData::ConfigV4 {
public:
  typedef PsddlPds::EvrData::ConfigV4 XtcType;
  typedef Psana::EvrData::ConfigV4 PsanaType;
  ConfigV4(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~ConfigV4();
  virtual uint32_t neventcodes() const;
  virtual uint32_t npulses() const;
  virtual uint32_t noutputs() const;
  virtual ndarray<Psana::EvrData::EventCodeV4, 1> eventcodes() const;
  virtual ndarray<Psana::EvrData::PulseConfigV3, 1> pulses() const;
  virtual ndarray<Psana::EvrData::OutputMap, 1> output_maps() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector<Psana::EvrData::EventCodeV4> _eventcodes_ndarray_storage_;
  unsigned _eventcodes_ndarray_shape_[1];
  std::vector<Psana::EvrData::PulseConfigV3> _pulses_ndarray_storage_;
  unsigned _pulses_ndarray_shape_[1];
  std::vector<Psana::EvrData::OutputMap> _output_maps_ndarray_storage_;
  unsigned _output_maps_ndarray_shape_[1];
};

Psana::EvrData::SequencerEntry pds_to_psana(PsddlPds::EvrData::SequencerEntry pds);


class SequencerConfigV1 : public Psana::EvrData::SequencerConfigV1 {
public:
  typedef PsddlPds::EvrData::SequencerConfigV1 XtcType;
  typedef Psana::EvrData::SequencerConfigV1 PsanaType;
  SequencerConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~SequencerConfigV1();
  virtual Psana::EvrData::SequencerConfigV1::Source sync_source() const;
  virtual Psana::EvrData::SequencerConfigV1::Source beam_source() const;
  virtual uint32_t length() const;
  virtual uint32_t cycles() const;
  virtual ndarray<Psana::EvrData::SequencerEntry, 1> entries() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector<Psana::EvrData::SequencerEntry> _entries_ndarray_storage_;
  unsigned _entries_ndarray_shape_[1];
};


class ConfigV5 : public Psana::EvrData::ConfigV5 {
public:
  typedef PsddlPds::EvrData::ConfigV5 XtcType;
  typedef Psana::EvrData::ConfigV5 PsanaType;
  ConfigV5(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~ConfigV5();
  virtual uint32_t neventcodes() const;
  virtual uint32_t npulses() const;
  virtual uint32_t noutputs() const;
  virtual ndarray<Psana::EvrData::EventCodeV5, 1> eventcodes() const;
  virtual ndarray<Psana::EvrData::PulseConfigV3, 1> pulses() const;
  virtual ndarray<Psana::EvrData::OutputMap, 1> output_maps() const;
  virtual const Psana::EvrData::SequencerConfigV1& seq_config() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector<Psana::EvrData::EventCodeV5> _eventcodes_ndarray_storage_;
  unsigned _eventcodes_ndarray_shape_[1];
  std::vector<Psana::EvrData::PulseConfigV3> _pulses_ndarray_storage_;
  unsigned _pulses_ndarray_shape_[1];
  std::vector<Psana::EvrData::OutputMap> _output_maps_ndarray_storage_;
  unsigned _output_maps_ndarray_shape_[1];
  psddl_pds2psana::EvrData::SequencerConfigV1 _seq_config;
};

Psana::EvrData::FIFOEvent pds_to_psana(PsddlPds::EvrData::FIFOEvent pds);


class DataV3 : public Psana::EvrData::DataV3 {
public:
  typedef PsddlPds::EvrData::DataV3 XtcType;
  typedef Psana::EvrData::DataV3 PsanaType;
  DataV3(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~DataV3();
  virtual uint32_t numFifoEvents() const;
  virtual ndarray<Psana::EvrData::FIFOEvent, 1> fifoEvents() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector<Psana::EvrData::FIFOEvent> _fifoEvents_ndarray_storage_;
  unsigned _fifoEvents_ndarray_shape_[1];
};

Psana::EvrData::IOChannel pds_to_psana(PsddlPds::EvrData::IOChannel pds);


class IOConfigV1 : public Psana::EvrData::IOConfigV1 {
public:
  typedef PsddlPds::EvrData::IOConfigV1 XtcType;
  typedef Psana::EvrData::IOConfigV1 PsanaType;
  IOConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~IOConfigV1();
  virtual uint16_t nchannels() const;
  virtual ndarray<Psana::EvrData::IOChannel, 1> channels() const;
  virtual Psana::EvrData::OutputMap::Conn conn() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector<Psana::EvrData::IOChannel> _channels_ndarray_storage_;
  unsigned _channels_ndarray_shape_[1];
};

} // namespace EvrData
} // namespace psddl_pds2psana
#endif // PSDDL_PDS2PSANA_EVR_DDL_H
