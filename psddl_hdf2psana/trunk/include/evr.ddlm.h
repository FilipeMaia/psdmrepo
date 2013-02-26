#ifndef PSDDL_HDF2PSANA_EVR_DDLM_H
#define PSDDL_HDF2PSANA_EVR_DDLM_H

#include <boost/shared_ptr.hpp>

#include "hdf5pp/Group.h"
#include "psddl_psana/evr.ddl.h"

namespace psddl_hdf2psana {
namespace EvrData {

class ConfigV5 : public Psana::EvrData::ConfigV5 {
public:

  typedef Psana::EvrData::ConfigV5 PsanaType;

  ConfigV5(hdf5pp::Group group);
  virtual ~ConfigV5();

  virtual uint32_t neventcodes() const;
  virtual uint32_t npulses() const;
  virtual uint32_t noutputs() const;
  virtual ndarray<const Psana::EvrData::EventCodeV5, 1> eventcodes() const;
  virtual ndarray<const Psana::EvrData::PulseConfigV3, 1> pulses() const;
  virtual ndarray<const Psana::EvrData::OutputMap, 1> output_maps() const;
  virtual const Psana::EvrData::SequencerConfigV1& seq_config() const;

private:

  struct eventcodes_data;
  struct pulses_data;
  struct output_maps_data;
  struct config_data;

  std::vector<Psana::EvrData::EventCodeV5> m_eventcodes;
  std::vector<Psana::EvrData::PulseConfigV3> m_pulses;
  std::vector<Psana::EvrData::OutputMap> m_output_maps;
  boost::shared_ptr<Psana::EvrData::SequencerConfigV1> m_seq_config;
  std::auto_ptr<config_data> m_config;

};


} // namespace EvrData
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_EVR.DDLM_H
