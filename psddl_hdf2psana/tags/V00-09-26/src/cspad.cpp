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

void
ns_ConfigV2_v0::dataset_config::init_attr_sections()
{
  // fill the list of active sections from ROI bits
  int ns = 0 ;
  for ( int q = 0; q < Psana::CsPad::MaxQuadsPerSensor ; ++ q ) {
    unsigned mask = (this->roiMask >> (8*q)) & 0xff;
    for ( int s = 0; s < Psana::CsPad::SectorsPerQuad ; ++ s ) {
      sections[q][s] = -1;
      if ( mask & (1<<s) ) sections[q][s] = ns++;
    }
  }
}

void
ns_ConfigV3_v0::dataset_config::init_attr_sections()
{
  // fill the list of active sections from ROI bits
  int ns = 0 ;
  for ( int q = 0; q < Psana::CsPad::MaxQuadsPerSensor ; ++ q ) {
    unsigned mask = (this->roiMask >> (8*q)) & 0xff;
    for ( int s = 0; s < Psana::CsPad::SectorsPerQuad ; ++ s ) {
      sections[q][s] = -1;
      if ( mask & (1<<s) ) sections[q][s] = ns++;
    }
  }
}

void
ns_ConfigV4_v0::dataset_config::init_attr_sections()
{
  // fill the list of active sections from ROI bits
  int ns = 0 ;
  for ( int q = 0; q < Psana::CsPad::MaxQuadsPerSensor ; ++ q ) {
    unsigned mask = (this->roiMask >> (8*q)) & 0xff;
    for ( int s = 0; s < Psana::CsPad::SectorsPerQuad ; ++ s ) {
      sections[q][s] = -1;
      if ( mask & (1<<s) ) sections[q][s] = ns++;
    }
  }
}

void
ns_ConfigV5_v0::dataset_config::init_attr_sections()
{
  // fill the list of active sections from ROI bits
  int ns = 0 ;
  for ( int q = 0; q < Psana::CsPad::MaxQuadsPerSensor ; ++ q ) {
    unsigned mask = (this->roiMask >> (8*q)) & 0xff;
    for ( int s = 0; s < Psana::CsPad::SectorsPerQuad ; ++ s ) {
      sections[q][s] = -1;
      if ( mask & (1<<s) ) sections[q][s] = ns++;
    }
  }
}


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

void make_datasets_DataV1_v0(const Psana::CsPad::DataV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  const unsigned nquads = obj.quads_shape()[0];
  const unsigned nsect = obj.quads(0).data().shape()[0];

  {
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<ns_ElementV1_v0::dataset_element>::stored_type(), nquads);
    hdf5pp::Utils::createDataset(group, "element", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
  {
    hsize_t dims[4] = {nquads, nsect, Psana::CsPad::ColumnsPerASIC, Psana::CsPad::MaxRowsPerASIC*2};
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<int16_t>::stored_type(), 4, dims);
    hdf5pp::Utils::createDataset(group, "data", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
  {
    hsize_t dims[2] = {nquads, nsect};
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<float>::stored_type(), 2, dims);
    hdf5pp::Utils::createDataset(group, "common_mode", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
}

void store_DataV1_v0(const Psana::CsPad::DataV1* obj, hdf5pp::Group group, long index, bool append)
{
  if (not obj) {
    if (append) {
      hdf5pp::Utils::resizeDataset(group, "element", index < 0 ? index : index + 1);
      hdf5pp::Utils::resizeDataset(group, "data", index < 0 ? index : index + 1);
      hdf5pp::Utils::resizeDataset(group, "common_mode", index < 0 ? index : index + 1);
    }
    return;
  }

  const unsigned nquads = obj->quads_shape()[0];
  const unsigned nsect = obj->quads(0).data().shape()[0];

  {
    ndarray<ns_ElementV1_v0::dataset_element, 1> data = make_ndarray<ns_ElementV1_v0::dataset_element>(nquads);
    for (unsigned i = 0; i != nquads; ++ i) {
      data[i] = ns_ElementV1_v0::dataset_element(obj->quads(i));
    }
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "element", data, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "element", data);
    }
  }
  {
    ndarray<int16_t, 4> data = make_ndarray<int16_t>(nquads, nsect, Psana::CsPad::ColumnsPerASIC, Psana::CsPad::MaxRowsPerASIC*2);
    for (unsigned i = 0; i != nquads; ++ i) {
      const ndarray<const int16_t, 3>& small = obj->quads(i).data();
      std::copy(small.begin(), small.end(), &data[i][0][0][0]);
    }
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "data", data, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "data", data);
    }
  }
  {
    ndarray<float, 2> data = make_ndarray<float>(nquads, nsect);
    for (unsigned i = 0; i != nquads; ++ i) {
      const Psana::CsPad::ElementV1& quad = obj->quads(i);
      for (unsigned s = 0; s != nsect; ++ s) {
        data[i][s] = quad.common_mode(i);
      }
    }
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "common_mode", data, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "common_mode", data);
    }
  }
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

void make_datasets_DataV2_v0(const Psana::CsPad::DataV2& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  const unsigned nquads = obj.quads_shape()[0];
  unsigned nsect = 0;
  for (unsigned q = 0; q != nquads; ++ q) {
    nsect += obj.quads(q).data().shape()[0];
  }

  {
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<ns_ElementV1_v0::dataset_element>::stored_type(), nquads);
    hdf5pp::Utils::createDataset(group, "element", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
  {
    hsize_t dims[3] = {nsect, Psana::CsPad::ColumnsPerASIC, Psana::CsPad::MaxRowsPerASIC*2};
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<int16_t>::stored_type(), 3, dims);
    hdf5pp::Utils::createDataset(group, "data", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
  {
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<float>::stored_type(), nsect);
    hdf5pp::Utils::createDataset(group, "common_mode", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
}

void store_DataV2_v0(const Psana::CsPad::DataV2* obj, hdf5pp::Group group, long index, bool append)
{
  if (not obj) {
    if (append) {
      hdf5pp::Utils::resizeDataset(group, "element", index < 0 ? index : index + 1);
      hdf5pp::Utils::resizeDataset(group, "data", index < 0 ? index : index + 1);
      hdf5pp::Utils::resizeDataset(group, "common_mode", index < 0 ? index : index + 1);
    }
    return;
  }

  const unsigned nquads = obj->quads_shape()[0];
  unsigned nsect = 0;
  for (unsigned q = 0; q != nquads; ++ q) {
    nsect += obj->quads(q).data().shape()[0];
  }

  {
    ndarray<ns_ElementV2_v0::dataset_element, 1> data = make_ndarray<ns_ElementV2_v0::dataset_element>(nquads);
    for (unsigned i = 0; i != nquads; ++ i) {
      data[i] = ns_ElementV2_v0::dataset_element(obj->quads(i));
    }
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "element", data, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "element", data);
    }
  }
  {
    ndarray<int16_t, 3> data = make_ndarray<int16_t>(nsect, Psana::CsPad::ColumnsPerASIC, Psana::CsPad::MaxRowsPerASIC*2);
    unsigned s = 0;
    for (unsigned i = 0; i != nquads; ++ i) {
      const ndarray<const int16_t, 3>& small = obj->quads(i).data();
      std::copy(small.begin(), small.end(), &data[s][0][0]);
      s += small.shape()[0];
    }
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "data", data, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "data", data);
    }
  }
  {
    ndarray<float, 1> data = make_ndarray<float>(nsect);
    unsigned s = 0;
    for (unsigned i = 0; i != nquads; ++ i) {
      const Psana::CsPad::ElementV2& quad = obj->quads(i);
      const unsigned ns = obj->quads(i).data().shape()[0];
      for (unsigned is = 0; is != ns; ++ is) {
        data[s++] = quad.common_mode(is);
      }
    }
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "common_mode", data, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "common_mode", data);
    }
  }
}

template class DataV2_v0<Psana::CsPad::ConfigV2>;
template class DataV2_v0<Psana::CsPad::ConfigV3>;
template class DataV2_v0<Psana::CsPad::ConfigV4>;
template class DataV2_v0<Psana::CsPad::ConfigV5>;

} // namespace CsPad
} // namespace psddl_hdf2psana
