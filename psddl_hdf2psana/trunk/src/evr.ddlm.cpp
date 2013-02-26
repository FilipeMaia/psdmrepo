#include "psddl_hdf2psana/evr.ddlm.h"

#include <boost/make_shared.hpp>
#include <iostream>

#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/VlenType.h"

namespace psddl_hdf2psana {
namespace EvrData {

struct ConfigV5::eventcodes_data {

  enum { DescSize = 16 };

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  operator Psana::EvrData::EventCodeV5() const;
  operator boost::shared_ptr<Psana::EvrData::EventCodeV5>() const;

  uint16_t  code;
  uint8_t   isReadout;
  uint8_t   isCommand;
  uint8_t   isLatch;
  uint32_t  reportDelay;
  uint32_t  reportWidth;
  uint32_t  releaseCode;
  uint32_t  maskTrigger;
  uint32_t  maskSet;
  uint32_t  maskClear;
  char*     desc;
};

hdf5pp::Type
ConfigV5::eventcodes_data::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ConfigV5::eventcodes_data::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType< ConfigV5::eventcodes_data >() ;
  type.insert_native<uint16_t>( "code", offsetof(ConfigV5::eventcodes_data,code) ) ;
  type.insert_native<uint8_t>( "isReadout", offsetof(ConfigV5::eventcodes_data,isReadout) ) ;
  type.insert_native<uint8_t>( "isCommand", offsetof(ConfigV5::eventcodes_data,isCommand) ) ;
  type.insert_native<uint8_t>( "isLatch", offsetof(ConfigV5::eventcodes_data,isLatch) ) ;
  type.insert_native<uint32_t>( "reportDelay", offsetof(ConfigV5::eventcodes_data,reportDelay) ) ;
  type.insert_native<uint32_t>( "reportWidth", offsetof(ConfigV5::eventcodes_data,reportWidth) ) ;
  type.insert_native<uint32_t>( "releaseCode", offsetof(ConfigV5::eventcodes_data,releaseCode) ) ;
  type.insert_native<uint32_t>( "maskTrigger", offsetof(ConfigV5::eventcodes_data,maskTrigger) ) ;
  type.insert_native<uint32_t>( "maskSet", offsetof(ConfigV5::eventcodes_data,maskSet) ) ;
  type.insert_native<uint32_t>( "maskClear", offsetof(ConfigV5::eventcodes_data,maskClear) ) ;
  type.insert_native<const char*>( "desc", offsetof(ConfigV5::eventcodes_data,desc) ) ;

  return type ;
}

ConfigV5::eventcodes_data::operator Psana::EvrData::EventCodeV5() const
{
  char desc_[DescSize];
  int i = 0;
  for ( ; i < DescSize-1 and desc[i]; ++ i) desc_[i] = desc[i];
  for ( ; i < DescSize; ++ i) desc_[i] = '\0';
  return Psana::EvrData::EventCodeV5(code, isReadout, isCommand, isLatch,
            reportDelay, reportWidth, maskTrigger, maskSet, maskClear, desc_);
}

ConfigV5::eventcodes_data::operator boost::shared_ptr<Psana::EvrData::EventCodeV5>() const
{
  return boost::shared_ptr<Psana::EvrData::EventCodeV5>(new Psana::EvrData::EventCodeV5(*this));
}


struct ConfigV5::pulses_data {

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  operator Psana::EvrData::PulseConfigV3() const;
  operator boost::shared_ptr<Psana::EvrData::PulseConfigV3>() const;

  uint16_t  pulseId;
  uint16_t  polarity;
  uint32_t  prescale;
  uint32_t  delay;
  uint32_t  width;
};

hdf5pp::Type
ConfigV5::pulses_data::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ConfigV5::pulses_data::native_type()
{
  hdf5pp::CompoundType pulseType = hdf5pp::CompoundType::compoundType< ConfigV5::pulses_data >() ;
  pulseType.insert_native<uint16_t>( "pulseId", offsetof(ConfigV5::pulses_data,pulseId) ) ;
  pulseType.insert_native<uint16_t>( "polarity", offsetof(ConfigV5::pulses_data,polarity) ) ;
  pulseType.insert_native<uint32_t>( "prescale", offsetof(ConfigV5::pulses_data,prescale) ) ;
  pulseType.insert_native<uint32_t>( "delay", offsetof(ConfigV5::pulses_data,delay) ) ;
  pulseType.insert_native<uint32_t>( "width", offsetof(ConfigV5::pulses_data,width) ) ;

  return pulseType ;
}

ConfigV5::pulses_data::operator Psana::EvrData::PulseConfigV3() const
{
  return Psana::EvrData::PulseConfigV3(pulseId, polarity, prescale, delay, width);
}

ConfigV5::pulses_data::operator boost::shared_ptr<Psana::EvrData::PulseConfigV3>() const
{
  return boost::make_shared<Psana::EvrData::PulseConfigV3>(pulseId, polarity, prescale, delay, width);
}

struct ConfigV5::output_maps_data {

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  operator Psana::EvrData::OutputMap() const;
  operator boost::shared_ptr<Psana::EvrData::OutputMap>() const;

  int16_t source ;
  int16_t source_id ;
  int16_t conn ;
  int16_t conn_id ;
};

hdf5pp::Type
ConfigV5::output_maps_data::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ConfigV5::output_maps_data::native_type()
{
  hdf5pp::EnumType<int16_t> srcEnumType = hdf5pp::EnumType<int16_t>::enumType() ;
  srcEnumType.insert ( "Pulse", Psana::EvrData::OutputMap::Pulse ) ;
  srcEnumType.insert ( "DBus", Psana::EvrData::OutputMap::DBus ) ;
  srcEnumType.insert ( "Prescaler", Psana::EvrData::OutputMap::Prescaler ) ;
  srcEnumType.insert ( "Force_High", Psana::EvrData::OutputMap::Force_High ) ;
  srcEnumType.insert ( "Force_Low", Psana::EvrData::OutputMap::Force_Low ) ;

  hdf5pp::EnumType<int16_t> connEnumType = hdf5pp::EnumType<int16_t>::enumType() ;
  connEnumType.insert ( "FrontPanel", Psana::EvrData::OutputMap::FrontPanel ) ;
  connEnumType.insert ( "UnivIO", Psana::EvrData::OutputMap::UnivIO ) ;

  hdf5pp::CompoundType mapType = hdf5pp::CompoundType::compoundType< ConfigV5::output_maps_data >() ;
  mapType.insert( "source", offsetof(ConfigV5::output_maps_data,source), srcEnumType ) ;
  mapType.insert_native<int16_t>( "source_id", offsetof(ConfigV5::output_maps_data,source_id) ) ;
  mapType.insert( "conn", offsetof(ConfigV5::output_maps_data,conn), connEnumType ) ;
  mapType.insert_native<int16_t>( "conn_id", offsetof(ConfigV5::output_maps_data,conn_id) ) ;

  return mapType ;
}

ConfigV5::output_maps_data::operator Psana::EvrData::OutputMap() const
{
  return Psana::EvrData::OutputMap(Psana::EvrData::OutputMap::Source(source),
      source_id, Psana::EvrData::OutputMap::Conn(conn), conn_id);
}

ConfigV5::output_maps_data::operator boost::shared_ptr<Psana::EvrData::OutputMap>() const
{
  return boost::make_shared<Psana::EvrData::OutputMap>(Psana::EvrData::OutputMap::Source(source),
      source_id, Psana::EvrData::OutputMap::Conn(conn), conn_id);
}

class SequencerConfigV1 : public Psana::EvrData::SequencerConfigV1 {
public:
  SequencerConfigV1(hdf5pp::Group group);

  virtual Psana::EvrData::SequencerConfigV1::Source sync_source() const;
  virtual Psana::EvrData::SequencerConfigV1::Source beam_source() const;
  virtual uint32_t length() const;
  virtual uint32_t cycles() const;
  virtual ndarray<const Psana::EvrData::SequencerEntry, 1> entries() const;

private:
  struct seq_config_entry_data;
  struct seq_config_data;

  std::auto_ptr<seq_config_data> m_seq_config;
  std::vector<Psana::EvrData::SequencerEntry> m_entries;
};

struct SequencerConfigV1::seq_config_entry_data {

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  operator Psana::EvrData::SequencerEntry() const;
  operator boost::shared_ptr<Psana::EvrData::SequencerEntry>() const;

  uint32_t eventcode;
  uint32_t delay;
};

hdf5pp::Type
SequencerConfigV1::seq_config_entry_data::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
SequencerConfigV1::seq_config_entry_data::native_type()
{
  // SequencerEntry type
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType< SequencerConfigV1::seq_config_entry_data >() ;
  type.insert_native<uint32_t>( "eventcode", offsetof(SequencerConfigV1::seq_config_entry_data, eventcode) ) ;
  type.insert_native<uint32_t>( "delay", offsetof(SequencerConfigV1::seq_config_entry_data, delay) ) ;
  return type ;
}

SequencerConfigV1::seq_config_entry_data::operator Psana::EvrData::SequencerEntry() const
{
  return Psana::EvrData::SequencerEntry(eventcode, delay);
}

SequencerConfigV1::seq_config_entry_data::operator boost::shared_ptr<Psana::EvrData::SequencerEntry>() const
{
  return boost::make_shared<Psana::EvrData::SequencerEntry>(eventcode, delay);
}

struct SequencerConfigV1::seq_config_data {

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  uint16_t sync_source;
  uint16_t beam_source;
  uint32_t cycles;
  uint32_t length;
  size_t nentries;
  SequencerConfigV1::seq_config_entry_data* entries;
};

hdf5pp::Type
SequencerConfigV1::seq_config_data::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
SequencerConfigV1::seq_config_data::native_type()
{
  // variable-size type for info array
  hdf5pp::Type entriesType = hdf5pp::VlenType::vlenType(SequencerConfigV1::seq_config_entry_data::native_type());

  hdf5pp::EnumType<int16_t> srcEnumType = hdf5pp::EnumType<int16_t>::enumType() ;
  srcEnumType.insert ( "r120Hz", Psana::EvrData::SequencerConfigV1::r120Hz ) ;
  srcEnumType.insert ( "r60Hz", Psana::EvrData::SequencerConfigV1::r60Hz ) ;
  srcEnumType.insert ( "r30Hz", Psana::EvrData::SequencerConfigV1::r30Hz ) ;
  srcEnumType.insert ( "r10Hz", Psana::EvrData::SequencerConfigV1::r10Hz ) ;
  srcEnumType.insert ( "r5Hz", Psana::EvrData::SequencerConfigV1::r5Hz ) ;
  srcEnumType.insert ( "r1Hz", Psana::EvrData::SequencerConfigV1::r1Hz ) ;
  srcEnumType.insert ( "r0_5Hz", Psana::EvrData::SequencerConfigV1::r0_5Hz ) ;
  srcEnumType.insert ( "Disable", Psana::EvrData::SequencerConfigV1::Disable ) ;

  // SequencerConfigV1 type
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType< SequencerConfigV1::seq_config_data >() ;
  type.insert( "sync_source", offsetof(SequencerConfigV1::seq_config_data, sync_source), srcEnumType ) ;
  type.insert( "beam_source", offsetof(SequencerConfigV1::seq_config_data, beam_source), srcEnumType ) ;
  type.insert_native<uint32_t>( "cycles", offsetof(SequencerConfigV1::seq_config_data, cycles) ) ;
  type.insert_native<uint32_t>( "length", offsetof(SequencerConfigV1::seq_config_data, length) ) ;
  type.insert( "entries", offsetof(SequencerConfigV1::seq_config_data, nentries), entriesType ) ;

  return type ;
}

SequencerConfigV1::SequencerConfigV1(hdf5pp::Group group)
{
  {
    hdf5pp::DataSet<seq_config_data> ds = group.openDataSet<seq_config_data>("seq_config");
    hdf5pp::DataSpace file_dsp = ds.dataSpace();
    m_seq_config.reset(new seq_config_data);
    ds.read(hdf5pp::DataSpace::makeScalar(), file_dsp, m_seq_config.get(), seq_config_data::native_type());
  }

  m_entries.reserve(m_seq_config->nentries);
  for (size_t i = 0 ; i != m_seq_config->nentries; ++ i) {
    m_entries.push_back(m_seq_config->entries[i]);
  }
}

Psana::EvrData::SequencerConfigV1::Source
SequencerConfigV1::sync_source() const
{
  return EvrData::SequencerConfigV1::Source(m_seq_config->sync_source);
}

Psana::EvrData::SequencerConfigV1::Source
SequencerConfigV1::beam_source() const
{
  return EvrData::SequencerConfigV1::Source(m_seq_config->beam_source);
}

uint32_t
SequencerConfigV1::length() const
{
  return m_seq_config->length;
}

uint32_t
SequencerConfigV1::cycles() const
{
  return m_seq_config->cycles;
}

ndarray<const Psana::EvrData::SequencerEntry, 1>
SequencerConfigV1::entries() const
{
  return make_ndarray(&m_entries.front(), m_entries.size());
}

struct ConfigV5::config_data {

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  uint32_t neventcodes;
  uint32_t npulses;
  uint32_t noutputs;
};

hdf5pp::Type
ConfigV5::config_data::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ConfigV5::config_data::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<ConfigV5::config_data>() ;
  confType.insert_native<uint32_t>( "neventcodes", offsetof(ConfigV5::config_data,neventcodes) ) ;
  confType.insert_native<uint32_t>( "npulses", offsetof(ConfigV5::config_data,npulses) ) ;
  confType.insert_native<uint32_t>( "noutputs", offsetof(ConfigV5::config_data,noutputs) ) ;
  return confType ;
}

ConfigV5::ConfigV5(hdf5pp::Group group)
{
  {
    hdf5pp::DataSet<eventcodes_data> ds = group.openDataSet<eventcodes_data>("eventcodes");
    hdf5pp::DataSpace file_dsp = ds.dataSpace();
    size_t size = file_dsp.size();
    if (size > 0) {
      std::vector<eventcodes_data> data(size);
      hdf5pp::DataSpace mem_dsp = hdf5pp::DataSpace::makeSimple(size, size);
      ds.read(mem_dsp, file_dsp, &data.front(), eventcodes_data::native_type());
      m_eventcodes.reserve(size);
      for(size_t i = 0; i != size; ++ i) {
        m_eventcodes.push_back(data[i]);
      }
      ds.vlen_reclaim(mem_dsp, &data.front(), eventcodes_data::native_type());
    }
  }
  {
    hdf5pp::DataSet<pulses_data> ds = group.openDataSet<pulses_data>("pulses");
    hdf5pp::DataSpace file_dsp = ds.dataSpace();
    size_t size = file_dsp.size();
    if (size > 0) {
      std::vector<pulses_data> data(size);
      hdf5pp::DataSpace mem_dsp = hdf5pp::DataSpace::makeSimple(size, size);
      ds.read(mem_dsp, file_dsp, &data.front(), pulses_data::native_type());
      m_pulses.reserve(size);
      for(size_t i = 0; i != size; ++ i) {
        m_pulses.push_back(data[i]);
      }
      ds.vlen_reclaim(mem_dsp, &data.front(), pulses_data::native_type());
    }
  }
  {
    hdf5pp::DataSet<output_maps_data> ds = group.openDataSet<output_maps_data>("output_maps");
    hdf5pp::DataSpace file_dsp = ds.dataSpace();
    size_t size = file_dsp.size();
    if (size > 0) {
      std::vector<output_maps_data> data(size);
      hdf5pp::DataSpace mem_dsp = hdf5pp::DataSpace::makeSimple(size, size);
      ds.read(mem_dsp, file_dsp, &data.front(), output_maps_data::native_type());
      m_output_maps.reserve(size);
      for(size_t i = 0; i != size; ++ i) {
        m_output_maps.push_back(data[i]);
      }
      ds.vlen_reclaim(mem_dsp, &data.front(), output_maps_data::native_type());
    }
  }

  m_seq_config.reset(new SequencerConfigV1(group));

  {
    hdf5pp::DataSet<config_data> ds = group.openDataSet<config_data>("config");
    hdf5pp::DataSpace file_dsp = ds.dataSpace();
    m_config.reset(new config_data);
    ds.read(hdf5pp::DataSpace::makeScalar(), file_dsp, m_config.get(), config_data::native_type());
  }
}

ConfigV5::~ConfigV5()
{
}

uint32_t
ConfigV5::neventcodes() const
{
  return m_config->neventcodes;
}

uint32_t
ConfigV5::npulses() const
{
  return m_config->npulses;
}

uint32_t
ConfigV5::noutputs() const
{
  return m_config->noutputs;
}

ndarray<const Psana::EvrData::EventCodeV5, 1>
ConfigV5::eventcodes() const
{
  return make_ndarray(&m_eventcodes.front(), m_eventcodes.size());
}

ndarray<const Psana::EvrData::PulseConfigV3, 1>
ConfigV5::pulses() const
{
  return make_ndarray(&m_pulses.front(), m_pulses.size());
}

ndarray<const Psana::EvrData::OutputMap, 1>
ConfigV5::output_maps() const
{
  return make_ndarray(&m_output_maps.front(), m_output_maps.size());
}

const Psana::EvrData::SequencerConfigV1&
ConfigV5::seq_config() const
{
  return *m_seq_config;
}

} // namespace EvrData
} // namespace psddl_hdf2psana
