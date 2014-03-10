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
#include "psddl_hdf2psana/pnccd.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iterator>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/VlenType.h"
#include "hdf5pp/Utils.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  struct Compare_ts {
    bool operator()(const psddl_hdf2psana::PNCCD::ns_FrameV1_v0::dataset_data& lhs,
        const psddl_hdf2psana::PNCCD::ns_FrameV1_v0::dataset_data& rhs) const
    {
      if (lhs.timeStampHi < rhs.timeStampHi) return true;
      if (lhs.timeStampHi > rhs.timeStampHi) return false;
      if (lhs.timeStampLo < rhs.timeStampLo) return true;
      return false;
    }
  };

}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------


namespace psddl_hdf2psana {
namespace PNCCD {


hdf5pp::Type ns_FrameV1_v0_dataset_data_stored_type()
{
  typedef ns_FrameV1_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("specialWord", offsetof(DsType, specialWord), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("frameNumber", offsetof(DsType, frameNumber), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("timeStampHi", offsetof(DsType, timeStampHi), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("timeStampLo", offsetof(DsType, timeStampLo), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

hdf5pp::Type ns_FrameV1_v0::dataset_data::stored_type()
{
  static hdf5pp::Type type = ns_FrameV1_v0_dataset_data_stored_type();
  return type;
}

hdf5pp::Type ns_FrameV1_v0_dataset_data_native_type()
{
  typedef ns_FrameV1_v0::dataset_data DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("specialWord", offsetof(DsType, specialWord), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("frameNumber", offsetof(DsType, frameNumber), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("timeStampHi", offsetof(DsType, timeStampHi), hdf5pp::TypeTraits<uint32_t>::native_type());
  type.insert("timeStampLo", offsetof(DsType, timeStampLo), hdf5pp::TypeTraits<uint32_t>::native_type());
  return type;
}

hdf5pp::Type ns_FrameV1_v0::dataset_data::native_type()
{
  static hdf5pp::Type type = ns_FrameV1_v0_dataset_data_native_type();
  return type;
}

ns_FrameV1_v0::dataset_data::dataset_data(const Psana::PNCCD::FrameV1& psanaobj)
  : specialWord(psanaobj.specialWord())
  , frameNumber(psanaobj.frameNumber())
  , timeStampHi(psanaobj.timeStampHi())
  , timeStampLo(psanaobj.timeStampLo())
{
}

uint32_t
FrameV1_v0::specialWord() const {
  return uint32_t(m_ds_data.specialWord);
}

uint32_t
FrameV1_v0::frameNumber() const {
  return uint32_t(m_ds_data.frameNumber);
}

uint32_t
FrameV1_v0::timeStampHi() const {
  return uint32_t(m_ds_data.timeStampHi);
}

uint32_t
FrameV1_v0::timeStampLo() const {
  return uint32_t(m_ds_data.timeStampLo);
}

ndarray<const uint16_t, 2>
FrameV1_v0::data() const {
  return make_ndarray(m_ds_frameData.data_ptr(), 512, 512);
}

ndarray<const uint16_t, 1>
FrameV1_v0::_data() const
{
  return m_ds_frameData;
}

/** Number of frames is determined by numLinks() method. */
template <typename Config>
const Psana::PNCCD::FrameV1&
FramesV1_v0<Config>::frame(uint32_t i0) const
{
  if (m_frames.empty()) read_frames();
  return m_frames[i0];
}

/** Method which returns the shape (dimensions) of the data returned by frame() method. */
template <typename Config>
std::vector<int>
FramesV1_v0<Config>::frame_shape() const
{
  if (m_frames.empty()) read_frames();
  return std::vector<int>(1, m_frames.shape()[0]);
}

template <typename Config>
void
FramesV1_v0<Config>::read_frames() const
{
  ndarray<const ns_FrameV1_v0::dataset_data, 1> mdata = hdf5pp::Utils::readNdarray<ns_FrameV1_v0::dataset_data, 1>(m_group, "frame", m_idx);
  ndarray<const uint16_t, 2> frdata = hdf5pp::Utils::readNdarray<uint16_t, 2>(m_group, "data", m_idx);

  unsigned const nlinks = mdata.shape()[0];

  m_frames = make_ndarray<FrameV1_v0>(nlinks);
  for (unsigned int i = 0; i != nlinks; ++ i) {
    m_frames[i] = FrameV1_v0(mdata[i], frdata[i]);
  }
}

template class FramesV1_v0<Psana::PNCCD::ConfigV1>;
template class FramesV1_v0<Psana::PNCCD::ConfigV2>;

void make_datasets_FramesV1_v0(const Psana::PNCCD::FramesV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  const unsigned nLinks = obj.numLinks();
  const unsigned data_size = obj.frame(0)._data().shape()[0];
  {
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<ns_FrameV1_v0::dataset_data>::stored_type(), nLinks);
    hdf5pp::Utils::createDataset(group, "frame", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
  {
    hsize_t dims[2] = {nLinks, data_size};
    hdf5pp::Type dstype = hdf5pp::ArrayType::arrayType(hdf5pp::TypeTraits<uint16_t>::stored_type(), 2, dims);
    hdf5pp::Utils::createDataset(group, "data", dstype, chunkPolicy.chunkSize(dstype), chunkPolicy.chunkCacheSize(dstype), deflate, shuffle);
  }
}

void store_FramesV1_v0(const Psana::PNCCD::FramesV1* obj, hdf5pp::Group group, long index, bool append)
{
  if (not obj) {
    if (append) {
      hdf5pp::Utils::resizeDataset(group, "frame", index < 0 ? index : index + 1);
      hdf5pp::Utils::resizeDataset(group, "data", index < 0 ? index : index + 1);
    }
    return;
  }

  unsigned nLinks = obj->numLinks();
  ndarray<ns_FrameV1_v0::dataset_data, 1> frames_ds = make_ndarray<ns_FrameV1_v0::dataset_data>(nLinks);
  for (unsigned i = 0; i != nLinks; ++ i) {
    frames_ds[i] = ns_FrameV1_v0::dataset_data(obj->frame(i));
  }

  // need to make 2-d array, high dimension is the number of links,
  // low dimension is the size of data. We do not store images as images,
  // but as 1-d data
  unsigned data_size = obj->frame(0)._data().shape()[0];
  ndarray<uint16_t, 2> data = make_ndarray<uint16_t>(nLinks, data_size);

  // copy the data
  for (unsigned i = 0; i != nLinks; ++ i) {
    ndarray<const uint16_t, 1> small = obj->frame(i)._data();
    std::copy(small.begin(), small.end(), &data[i][0]);
  }

  if (append) {
    hdf5pp::Utils::storeNDArrayAt(group, "frame", frames_ds, index);
    hdf5pp::Utils::storeNDArrayAt(group, "data", data, index);
  } else {
    hdf5pp::Utils::storeNDArray(group, "frame", frames_ds);
    hdf5pp::Utils::storeNDArray(group, "data", data);
  }

}


/** Special values */
uint32_t
FullFrameV1_v0::specialWord() const
{
  if (m_frame.empty()) read_frame();
  return m_specialWord;
}

/** Frame number */
uint32_t
FullFrameV1_v0::frameNumber() const
{
  if (m_frame.empty()) read_frame();
  return m_frameNumber;
}

/** Most significant part of timestamp */
uint32_t
FullFrameV1_v0::timeStampHi() const
{
  if (m_frame.empty()) read_frame();
  return m_timeStampHi;
}

/** Least significant part of timestamp */
uint32_t
FullFrameV1_v0::timeStampLo() const
{
  if (m_frame.empty()) read_frame();
  return m_timeStampLo;
}

/** Full frame data, image size is 1024x1024. */
ndarray<const uint16_t, 2>
FullFrameV1_v0::data() const
{
  if (m_frame.empty()) read_frame();
  return m_frame;
}

void
FullFrameV1_v0::read_frame() const
{
  ndarray<const ns_FrameV1_v0::dataset_data, 1> mdata = hdf5pp::Utils::readNdarray<ns_FrameV1_v0::dataset_data, 1>(m_group, "frame", m_idx);
  ndarray<const uint16_t, 2> frdata = hdf5pp::Utils::readNdarray<uint16_t, 2>(m_group, "data", m_idx);

  unsigned const nlinks = mdata.shape()[0];
  if (nlinks != 4) {
    MsgLog("FullFrameV1_v0", error, "read_frame: number of links in PNCCD::FrameV1 is not equal 4: " << nlinks);
    return;
  }

  // copy metadata from the first frame, they all should be identical
  m_specialWord = mdata[0].specialWord;
  m_frameNumber = mdata[0].frameNumber;

  // take the lowest timestamp of four frames
  ndarray<const ns_FrameV1_v0::dataset_data, 1>::iterator it = std::min_element(mdata.begin(), mdata.end(), ::Compare_ts());
  m_timeStampHi = it->timeStampHi;
  m_timeStampLo = it->timeStampLo;

  // make large image out of four small images
  m_frame = make_ndarray<uint16_t>(1024, 1024);
  uint16_t* dest = &m_frame[0][0];
  const uint16_t* src0 = &frdata[0][0];
  const uint16_t* src3 = &frdata[3][0];
  for (int iY = 0; iY < 512; ++ iY, src0 += 512, src3 += 512) {
    dest = std::copy(src0, src0+512, dest);
    dest = std::copy(src3, src3+512, dest);
  }

  typedef std::reverse_iterator<const uint16_t*> RevIter;
  RevIter src1(&frdata[1][512*512]);
  RevIter src2(&frdata[2][512*512]);
  for (int iY = 0; iY < 512; ++ iY, src1 += 512, src2 += 512) {
    dest = std::copy(src1, src1+512, dest);
    dest = std::copy(src2, src2+512, dest);
  }
}

void make_datasets_FullFrameV1_v0(const Psana::PNCCD::FullFrameV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle)
{
  // we do not want to save FullFrame in HDF5
  MsgLog("PNCCD::make_datasets_FullFrameV1_v0", error, "type is not supported");
}

void store_FullFrameV1_v0(const Psana::PNCCD::FullFrameV1* obj, hdf5pp::Group group, long index, bool append)
{
  // we do not want to save FullFrame in HDF5
  MsgLog("PNCCD::store_FullFrameV1_v0", error, "type is not supported");
}

} // namespace PNCCD
} // namespace psddl_hdf2psana
