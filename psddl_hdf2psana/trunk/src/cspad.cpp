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
#include "psddl_hdf2psana/cspad.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/Utils.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "psddl_hdf2psana.CsPad";

}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

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
    std::fill(ds_cm.begin(), ds_cm.end(), 0.f);
  }

  const unsigned nelem = ds_element.size();
  m_elements = make_ndarray<DataV1_v0_Element>(nelem);
  for (unsigned i = 0; i != nelem; ++ i) {
    m_elements[i] = DataV1_v0_Element(ds_element[i], ds_data[i], ds_cm[i]);
  }
}

void store_DataV1_v0(const Psana::CsPad::DataV1& obj, hdf5pp::Group group, bool append)
{
    
}


template class DataV1_v0<Psana::CsPad::ConfigV1>;
template class DataV1_v0<Psana::CsPad::ConfigV2>;
template class DataV1_v0<Psana::CsPad::ConfigV3>;
template class DataV1_v0<Psana::CsPad::ConfigV4>;
template class DataV1_v0<Psana::CsPad::ConfigV5>;

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

  ndarray<int16_t, 3> ds_data = hdf5pp::Utils::readNdarray<int16_t, 3>(m_group, "data", m_idx);

  // common_mode dataset is optional.
  ndarray<float, 1> ds_cm;
  if (m_group.hasChild("common_mode")) {
    ds_cm = hdf5pp::Utils::readNdarray<float, 1>(m_group, "common_mode", m_idx);
  } else {
    // if dataset is missing make all-zeros array
    ds_cm = make_ndarray<float>(m_cfg->numQuads()*ASICsPerQuad/2);
    std::fill(ds_cm.begin(), ds_cm.end(), 0.f);
  }

  const unsigned nelem = ds_element.size();
  m_elements = make_ndarray<DataV2_v0_Element>(nelem);
  for (unsigned i = 0; i != nelem; ++ i) {

    // need range of indices of ds_data corresponding to this quadrant
    unsigned quad = ds_element[i].quad;

    // get number of segments before this quad and in this quad
    unsigned seg_before = 0;
    for (unsigned iq = 0; iq < quad; ++ iq) {
      seg_before += m_cfg->numAsicsStored(iq)/2;
    }
    unsigned seg_this = m_cfg->numAsicsStored(quad)/2;

    MsgLog(logger, debug, "DataV2_v0::read_elements: quad=" << quad << " seg_before=" << seg_before << " seg_this=" << seg_this);

    boost::shared_ptr<int16_t> data_ptr(ds_data.data_ptr(), &ds_data[seg_before][0][0]);
    ndarray<int16_t, 3> quad_data = make_ndarray(data_ptr, seg_this, ds_data.shape()[1], ds_data.shape()[2]);

    boost::shared_ptr<float> cm_ptr(ds_cm.data_ptr(), &ds_cm[seg_before]);
    ndarray<float, 1> cm_data = make_ndarray(cm_ptr, seg_this);

    m_elements[i] = DataV2_v0_Element(ds_element[i], quad_data, cm_data);
  }
}

void store_DataV2_v0(const Psana::CsPad::DataV2& obj, hdf5pp::Group group, bool append)
{
    
}

template class DataV2_v0<Psana::CsPad::ConfigV2>;
template class DataV2_v0<Psana::CsPad::ConfigV3>;
template class DataV2_v0<Psana::CsPad::ConfigV4>;
template class DataV2_v0<Psana::CsPad::ConfigV5>;

} // namespace CsPad
} // namespace psddl_hdf2psana
