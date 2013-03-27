//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Hand-written supporting types for DDL-HDF5 mapping.
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_hdf2psana/acqiris.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Utils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace psddl_hdf2psana {
namespace Acqiris {

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
