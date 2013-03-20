#include "psddl_hdf2psana/acqiris.ddlm.h"

#include "hdf5pp/CompoundType.h"
#include "hdf5pp/Utils.h"

namespace psddl_hdf2psana {
namespace Acqiris {




hdf5pp::Type ns_ConfigV1_v0_dataset_config_stored_type()
{
  typedef ns_ConfigV1_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("nbrConvertersPerChannel", offsetof(DsType, nbrConvertersPerChannel), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("channelMask", offsetof(DsType, channelMask), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("nbrBanks", offsetof(DsType, nbrBanks), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("nbrChannels", offsetof(DsType, nbrChannels), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_ConfigV1_v0::dataset_config::stored_type()
{
  static hdf5pp::Type type = ns_ConfigV1_v0_dataset_config_stored_type();
  return type;
}

hdf5pp::Type ns_ConfigV1_v0_dataset_config_native_type()
{
  typedef ns_ConfigV1_v0::dataset_config DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("nbrConvertersPerChannel", offsetof(DsType, nbrConvertersPerChannel), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("channelMask", offsetof(DsType, channelMask), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("nbrBanks", offsetof(DsType, nbrBanks), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("nbrChannels", offsetof(DsType, nbrChannels), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

hdf5pp::Type ns_ConfigV1_v0::dataset_config::native_type()
{
  static hdf5pp::Type type = ns_ConfigV1_v0_dataset_config_native_type();
  return type;
}
ns_ConfigV1_v0::dataset_config::dataset_config()
{
}
ns_ConfigV1_v0::dataset_config::~dataset_config()
{
}


uint32_t ConfigV1_v0::nbrConvertersPerChannel() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->nbrConvertersPerChannel);
}
uint32_t ConfigV1_v0::channelMask() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->channelMask);
}
uint32_t ConfigV1_v0::nbrBanks() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->nbrBanks);
}
const Psana::Acqiris::TrigV1& ConfigV1_v0::trig() const {
  if (not m_ds_trig) read_ds_trig();
  m_ds_storage_trig_trig = Psana::Acqiris::TrigV1(*m_ds_trig);
  return m_ds_storage_trig_trig;
}
const Psana::Acqiris::HorizV1& ConfigV1_v0::horiz() const {
  if (not m_ds_horiz) read_ds_horiz();
  m_ds_storage_horiz_horiz = Psana::Acqiris::HorizV1(*m_ds_horiz);
  return m_ds_storage_horiz_horiz;
}
ndarray<const Psana::Acqiris::VertV1, 1> ConfigV1_v0::vert() const {
  if (m_ds_vert.empty()) read_ds_vert();
  if (m_ds_storage_vert_vert.empty()) {
    ndarray<Psana::Acqiris::VertV1, 1> tmparr(m_ds_vert.shape());
    std::copy(m_ds_vert.begin(), m_ds_vert.end(), tmparr.begin());
    m_ds_storage_vert_vert = tmparr;
  }
  return m_ds_storage_vert_vert;
}

uint32_t ConfigV1_v0::nbrChannels() const {
  if (not m_ds_config) read_ds_config();
  return uint32_t(m_ds_config->nbrChannels);
}
void ConfigV1_v0::read_ds_config() const {
  m_ds_config = hdf5pp::Utils::readGroup<Acqiris::ns_ConfigV1_v0::dataset_config>(m_group, "config", m_idx);
}
void ConfigV1_v0::read_ds_horiz() const {
  m_ds_horiz = hdf5pp::Utils::readGroup<Acqiris::ns_HorizV1_v0::dataset_data>(m_group, "horiz", m_idx);
}
void ConfigV1_v0::read_ds_trig() const {
  m_ds_trig = hdf5pp::Utils::readGroup<Acqiris::ns_TrigV1_v0::dataset_data>(m_group, "trig", m_idx);
}

void ConfigV1_v0::read_ds_vert() const {
  m_ds_vert = hdf5pp::Utils::readNdarray<Acqiris::ns_VertV1_v0::dataset_data, 1>(m_group, "vert", m_idx);
}



template <typename Config>
const Psana::Acqiris::DataDescV1Elem&
DataDescV1_v0<Config>::data(uint32_t i0) const
{
  if (m_ds_data.empty()) read_ds_data();
  return m_ds_data[i0];
}


template <typename Config>
std::vector<int>
DataDescV1_v0<Config>::data_shape() const
{
  if (m_ds_data.empty()) read_ds_data();
  return std::vector<int>(1, m_ds_data.shape()[0]);
}

template <typename Config>
void
DataDescV1_v0<Config>::read_ds_data() const
{
  ndarray<const uint64_t, 2> tsint = hdf5pp::Utils::readNdarray<uint64_t, 2>(m_group, "timestamps", m_idx);
  ndarray<const int16_t, 3> waveforms = hdf5pp::Utils::readNdarray<int16_t, 3>(m_group, "waveforms", m_idx);

  const unsigned nch = tsint.shape()[0];
  const unsigned nseg = tsint.shape()[1];

  m_ds_data = make_ndarray<DataDescV1Elem_v0<Config> >(nch);
  for (unsigned ich = 0; ich != nch; ++ ich) {

    ndarray<Psana::Acqiris::TimestampV1, 1> ts = make_ndarray<Psana::Acqiris::TimestampV1>(nseg);

    for (unsigned is = 0; is != nseg; ++ is) {
      uint64_t ts64 = tsint[ich][is];
      uint32_t ts_low = ts64 & 0xffffffff;
      ts64 >>= 32;
      uint32_t ts_high = ts64 & 0xffffffff;
      ts[is] = Psana::Acqiris::TimestampV1(0.0, ts_low, ts_high);
    }

    m_ds_data[ich] = DataDescV1Elem_v0<Config>(ts, waveforms[ich], m_cfg);
  }
}

template class DataDescV1_v0<Psana::Acqiris::ConfigV1>;

} // namespace Acqiris
} // namespace psddl_hdf2psana
