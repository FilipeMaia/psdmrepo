#include "psddl_hdf2psana/cspad.ddlm.h"

#include <boost/make_shared.hpp>

#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/Utils.h"

namespace psddl_hdf2psana {
namespace CsPad {

template <typename Config>
const Psana::CsPad::ElementV1&
DataV1_v0<Config>::quads(uint32_t i0) const
{
  if (m_elements.empty()) read_elements();
  return m_elements[i0];
}

template <typename Config>
std::vector<int>
DataV1_v0<Config>::quads_shape() const
{
  return std::vector<int>(1, m_cfg->numQuads());
}

template <typename Config>
void
DataV1_v0<Config>::read_elements() const
{
  // read "element"
  ndarray<CsPad::ns_ElementV1_v0::dataset_element, 1> ds_element =
      hdf5pp::Utils::readNdarray<CsPad::ns_ElementV1_v0::dataset_element, 1>(m_group, "element", m_idx);

  ndarray<int16_t, 4> ds_data = hdf5pp::Utils::readNdarray<int16_t, 4>(m_group, "data", m_idx);

  // common_mode dataset is optional.
  ndarray<float, 2> ds_cm;
  if (m_group.hasChild("common_mode")) {
    ds_cm = hdf5pp::Utils::readNdarray<float, 2>(m_group, "common_mode", m_idx);
  } else {
    // if dataset is missing make all-zeros array
    ds_cm = make_ndarray<float>(m_cfg->numQuads(), m_cfg->numSect());
  }

  const unsigned nelem = ds_element.size();
  m_elements = make_ndarray<DataV1_v0_Element>(nelem);
  for (unsigned i = 0; i != nelem; ++ i) {
    m_elements[i] = DataV1_v0_Element(ds_element[i], ds_data[i], ds_cm[i]);
  }
}

template class DataV1_v0<Psana::CsPad::ConfigV1>;
template class DataV1_v0<Psana::CsPad::ConfigV2>;
template class DataV1_v0<Psana::CsPad::ConfigV3>;
template class DataV1_v0<Psana::CsPad::ConfigV4>;

template <typename Config>
const Psana::CsPad::ElementV2&
DataV2_v0<Config>::quads(uint32_t i0) const
{
  if (m_elements.empty()) read_elements();
  return m_elements[i0];
}

template <typename Config>
std::vector<int>
DataV2_v0<Config>::quads_shape() const
{
  return std::vector<int>(1, m_cfg->numQuads());
}

template <typename Config>
void
DataV2_v0<Config>::read_elements() const
{
  // read "element"
  ndarray<CsPad::ns_ElementV2_v0::dataset_element, 1> ds_element =
      hdf5pp::Utils::readNdarray<CsPad::ns_ElementV2_v0::dataset_element, 1>(m_group, "element", m_idx);

  ndarray<int16_t, 4> ds_data = hdf5pp::Utils::readNdarray<int16_t, 4>(m_group, "data", m_idx);

  // common_mode dataset is optional.
  ndarray<float, 2> ds_cm;
  if (m_group.hasChild("common_mode")) {
    ds_cm = hdf5pp::Utils::readNdarray<float, 2>(m_group, "common_mode", m_idx);
  } else {
    // if dataset is missing make all-zeros array
    ds_cm = make_ndarray<float>(m_cfg->numQuads(), m_cfg->numSect());
  }

  const unsigned nelem = ds_element.size();
  m_elements = make_ndarray<DataV2_v0_Element>(nelem);
  for (unsigned i = 0; i != nelem; ++ i) {
    m_elements[i] = DataV2_v0_Element(ds_element[i], ds_data[i], ds_cm[i]);
  }
}

template class DataV2_v0<Psana::CsPad::ConfigV2>;
template class DataV2_v0<Psana::CsPad::ConfigV3>;
template class DataV2_v0<Psana::CsPad::ConfigV4>;

} // namespace CsPad
} // namespace psddl_hdf2psana
