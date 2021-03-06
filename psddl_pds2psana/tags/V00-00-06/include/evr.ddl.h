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
  virtual const Psana::EvrData::PulseConfig& pulses(uint32_t i0) const;
  virtual const Psana::EvrData::OutputMap& output_maps(uint32_t i0) const;
  virtual std::vector<int> pulses_shape() const;
  virtual std::vector<int> output_maps_shape() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector< Psana::EvrData::PulseConfig > _pulses;
  std::vector< Psana::EvrData::OutputMap > _output_maps;
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
  virtual const Psana::EvrData::PulseConfig& pulses(uint32_t i0) const;
  virtual const Psana::EvrData::OutputMap& output_maps(uint32_t i0) const;
  virtual Psana::EvrData::ConfigV2::BeamCode beam() const;
  virtual Psana::EvrData::ConfigV2::RateCode rate() const;
  virtual std::vector<int> pulses_shape() const;
  virtual std::vector<int> output_maps_shape() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector< Psana::EvrData::PulseConfig > _pulses;
  std::vector< Psana::EvrData::OutputMap > _output_maps;
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
  virtual const Psana::EvrData::EventCodeV3& eventcodes(uint32_t i0) const;
  virtual const Psana::EvrData::PulseConfigV3& pulses(uint32_t i0) const;
  virtual const Psana::EvrData::OutputMap& output_maps(uint32_t i0) const;
  virtual std::vector<int> eventcodes_shape() const;
  virtual std::vector<int> pulses_shape() const;
  virtual std::vector<int> output_maps_shape() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector< Psana::EvrData::EventCodeV3 > _eventcodes;
  std::vector< Psana::EvrData::PulseConfigV3 > _pulses;
  std::vector< Psana::EvrData::OutputMap > _output_maps;
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
  virtual const Psana::EvrData::EventCodeV4& eventcodes(uint32_t i0) const;
  virtual const Psana::EvrData::PulseConfigV3& pulses(uint32_t i0) const;
  virtual const Psana::EvrData::OutputMap& output_maps(uint32_t i0) const;
  virtual std::vector<int> eventcodes_shape() const;
  virtual std::vector<int> pulses_shape() const;
  virtual std::vector<int> output_maps_shape() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector< Psana::EvrData::EventCodeV4 > _eventcodes;
  std::vector< Psana::EvrData::PulseConfigV3 > _pulses;
  std::vector< Psana::EvrData::OutputMap > _output_maps;
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
  virtual const Psana::EvrData::SequencerEntry& entries(uint32_t i0) const;
  virtual std::vector<int> entries_shape() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector< Psana::EvrData::SequencerEntry > _entries;
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
  virtual const Psana::EvrData::EventCodeV5& eventcodes(uint32_t i0) const;
  virtual const Psana::EvrData::PulseConfigV3& pulses(uint32_t i0) const;
  virtual const Psana::EvrData::OutputMap& output_maps(uint32_t i0) const;
  virtual const Psana::EvrData::SequencerConfigV1& seq_config() const;
  virtual std::vector<int> eventcodes_shape() const;
  virtual std::vector<int> pulses_shape() const;
  virtual std::vector<int> output_maps_shape() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector< Psana::EvrData::EventCodeV5 > _eventcodes;
  std::vector< Psana::EvrData::PulseConfigV3 > _pulses;
  std::vector< Psana::EvrData::OutputMap > _output_maps;
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
  virtual const Psana::EvrData::FIFOEvent& fifoEvents(uint32_t i0) const;
  virtual std::vector<int> fifoEvents_shape() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector< Psana::EvrData::FIFOEvent > _fifoEvents;
};


class IOChannel : public Psana::EvrData::IOChannel {
public:
  typedef PsddlPds::EvrData::IOChannel XtcType;
  typedef Psana::EvrData::IOChannel PsanaType;
  IOChannel(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~IOChannel();
  virtual const char* name() const;
  virtual uint32_t ninfo() const;
  virtual const Pds::DetInfo& infos(uint32_t i0) const;
  virtual std::vector<int> name_shape() const;
  virtual std::vector<int> infos_shape() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector< Pds::DetInfo > _info;
};


class IOConfigV1 : public Psana::EvrData::IOConfigV1 {
public:
  typedef PsddlPds::EvrData::IOConfigV1 XtcType;
  typedef Psana::EvrData::IOConfigV1 PsanaType;
  IOConfigV1(const boost::shared_ptr<const XtcType>& xtcPtr);
  virtual ~IOConfigV1();
  virtual uint16_t nchannels() const;
  virtual const Psana::EvrData::IOChannel& channels(uint32_t i0) const;
  virtual Psana::EvrData::OutputMap::Conn conn() const;
  virtual std::vector<int> channels_shape() const;
  const XtcType& _xtcObj() const { return *m_xtcObj; }
private:
  boost::shared_ptr<const XtcType> m_xtcObj;
  std::vector< psddl_pds2psana::EvrData::IOChannel > _channels;
};

} // namespace EvrData
} // namespace psddl_pds2psana
#endif // PSDDL_PDS2PSANA_EVR_DDL_H
