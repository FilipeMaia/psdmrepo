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
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/Utils.h"
#include "MsgLogger/MsgLogger.h"

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

void make_datasets_DataDescV1_v0(const Psana::Acqiris::DataDescV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  // this schema is too old, we'll not be writing this stuff anymore
  MsgLog("Acqiris::make_datasets_DataDescV1_v0", error, "schema is not supported");
}

void store_DataDescV1_v0(const Psana::Acqiris::DataDescV1* obj, hdf5pp::Group group, long index, bool append)
{
  // this schema is too old, we'll not be writing this stuff anymore
  MsgLog("Acqiris::store_DataDescV1_v0", error, "schema is not supported");
}


template class DataDescV1_v0<Psana::Acqiris::ConfigV1>;



hdf5pp::Type ns_DataDescV1Elem_v1_dataset_data_stored_type()
{
  typedef ns_DataDescV1Elem_v1::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("nbrSamplesInSeg", offsetof(DsType, nbrSamplesInSeg), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("indexFirstPoint", offsetof(DsType, indexFirstPoint), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("nbrSegments", offsetof(DsType, nbrSegments), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_DataDescV1Elem_v1::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_DataDescV1Elem_v1_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_DataDescV1Elem_v1_dataset_data_native_type()
{
  typedef ns_DataDescV1Elem_v1::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("nbrSamplesInSeg", offsetof(DsType, nbrSamplesInSeg), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("indexFirstPoint", offsetof(DsType, indexFirstPoint), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("nbrSegments", offsetof(DsType, nbrSegments), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_DataDescV1Elem_v1::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_DataDescV1Elem_v1_dataset_data_native_type();
  return type;
}

ns_DataDescV1Elem_v1::dataset_data::dataset_data(const Psana::Acqiris::DataDescV1Elem& psanaobj)
  : nbrSamplesInSeg(psanaobj.nbrSamplesInSeg())
  , indexFirstPoint(psanaobj.indexFirstPoint())
  , nbrSegments(psanaobj.nbrSegments())
{
}


template <typename Config>
const Psana::Acqiris::DataDescV1Elem&
DataDescV1_v1<Config>::data(uint32_t i0) const
{
  if (m_ds_data.empty()) read_ds_data();
  return m_ds_data[i0];
}


template <typename Config>
std::vector<int>
DataDescV1_v1<Config>::data_shape() const
{
  if (m_ds_data.empty()) read_ds_data();
  return std::vector<int>(1, m_ds_data.shape()[0]);
}

template <typename Config>
void
DataDescV1_v1<Config>::read_ds_data() const
{
  ndarray<ns_TimestampV1_v0::dataset_data, 2> tsint = hdf5pp::Utils::readNdarray<ns_TimestampV1_v0::dataset_data, 2>(m_group, "timestamps", m_idx);
  ndarray<const int16_t, 3> waveforms = hdf5pp::Utils::readNdarray<int16_t, 3>(m_group, "waveforms", m_idx);
  ndarray<ns_DataDescV1Elem_v1::dataset_data, 1> data = hdf5pp::Utils::readNdarray<ns_DataDescV1Elem_v1::dataset_data, 1>(m_group, "data", m_idx);

  const unsigned nch = tsint.shape()[0];
  const unsigned nseg = tsint.shape()[1];

  ndarray<Psana::Acqiris::TimestampV1, 2> ts(tsint.shape());
  m_ds_data = make_ndarray<DataDescV1Elem_v1<Config> >(nch);
  for (unsigned ich = 0; ich != nch; ++ ich) {

    // convert timestamps
    for (unsigned iseg = 0; iseg != nseg; ++ iseg) {
      uint64_t val = tsint[ich][iseg].value;
      ts[ich][iseg] = Psana::Acqiris::TimestampV1(tsint[ich][iseg].pos, val & 0xFFFFFFFF, val >> 32);
    }

    ndarray<Psana::Acqiris::TimestampV1, 1> chts = ts[ich];
    m_ds_data[ich] = DataDescV1Elem_v1<Config>(chts, waveforms[ich], data[ich]);
  }
}

void make_datasets_DataDescV1_v1(const Psana::Acqiris::DataDescV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  const unsigned nch = obj.data_shape()[0];
  const Psana::Acqiris::DataDescV1Elem& elem = obj.data(0);
  const ndarray<const int16_t, 2>& wf = elem.waveforms();
  const unsigned nseg = wf.shape()[0];
  const unsigned nsampl = wf.shape()[1];
  {
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<ns_DataDescV1Elem_v1::dataset_data>::stored_type(), nch);
    hdf5pp::Utils::createDataset(group, "data", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
  {
    hsize_t dim[2] = {nch, nseg};
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<ns_TimestampV1_v0::dataset_data>::stored_type(), 2, dim);
    hdf5pp::Utils::createDataset(group, "timestamps", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
  {
    hsize_t dim[3] = {nch, nseg, nsampl};
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<int16_t>::stored_type(), 3, dim);
    hdf5pp::Utils::createDataset(group, "waveforms", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
}

void store_DataDescV1_v1(const Psana::Acqiris::DataDescV1* obj, hdf5pp::Group group, long index, bool append)
{
  if (not obj) {
    if (append) {
      hdf5pp::Utils::resizeDataset(group, "data", index < 0 ? index : index + 1);
      hdf5pp::Utils::resizeDataset(group, "timestamps", index < 0 ? index : index + 1);
      hdf5pp::Utils::resizeDataset(group, "waveforms", index < 0 ? index : index + 1);
    }
    return;
  }

  const unsigned nch = obj->data_shape()[0];
  const Psana::Acqiris::DataDescV1Elem& elem = obj->data(0);
  const ndarray<const int16_t, 2>& wf = elem.waveforms();
  const unsigned nseg = wf.shape()[0];
  const unsigned nsampl = wf.shape()[1];

  {
    ndarray<ns_DataDescV1Elem_v1::dataset_data, 1> data = make_ndarray<ns_DataDescV1Elem_v1::dataset_data>(nch);
    for (unsigned i = 0; i != nch; ++ i) {
      data[i] = ns_DataDescV1Elem_v1::dataset_data(obj->data(i));
    }
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "data", data, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "data", data);
    }
  }
  {
    ndarray<ns_TimestampV1_v0::dataset_data, 2> data = make_ndarray<ns_TimestampV1_v0::dataset_data>(nch, nseg);
    for (unsigned i = 0; i != nch; ++ i) {
      const ndarray<const Psana::Acqiris::TimestampV1, 1>& small = obj->data(i).timestamp();
      std::copy(small.begin(), small.end(), &data[i][0]);
    }
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "timestamps", data, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "timestamps", data);
    }
  }
  {
    ndarray<int16_t, 3> data = make_ndarray<int16_t>(nch, nseg, nsampl);
    for (unsigned i = 0; i != nch; ++ i) {
      const ndarray<const int16_t, 2>& small = obj->data(i).waveforms();
      std::copy(small.begin(), small.end(), &data[i][0][0]);
    }
    if (append) {
      hdf5pp::Utils::storeNDArrayAt(group, "waveforms", data, index);
    } else {
      hdf5pp::Utils::storeNDArray(group, "waveforms", data);
    }
  }
  
}

template class DataDescV1_v1<Psana::Acqiris::ConfigV1>;




} // namespace Acqiris
} // namespace psddl_hdf2psana
