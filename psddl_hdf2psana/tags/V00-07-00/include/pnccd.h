#ifndef PSDDL_HDF2PSANA_PNCCD_H
#define PSDDL_HDF2PSANA_PNCCD_H 1

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Hand-written supporting types for DDL-HDF5 mapping.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "psddl_hdf2psana/ChunkPolicy.h"
#include "psddl_psana/pnccd.ddl.h"

namespace psddl_hdf2psana {
namespace PNCCD {


namespace ns_FrameV1_v0 {
struct dataset_data {

  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_data() {}
  dataset_data(const Psana::PNCCD::FrameV1& psanaobj);

  uint32_t specialWord; 
  uint32_t frameNumber; 
  uint32_t timeStampHi; 
  uint32_t timeStampLo; 

};
}


class FrameV1_v0 : public Psana::PNCCD::FrameV1 {
public:
  typedef Psana::PNCCD::FrameV1 PsanaType;

  FrameV1_v0() {}
  FrameV1_v0(const PNCCD::ns_FrameV1_v0::dataset_data& ds_data,
      ndarray<const uint16_t, 1> ds_frameData)
    : m_ds_data(ds_data), m_ds_frameData(ds_frameData) {}

  virtual ~FrameV1_v0() {}

  virtual uint32_t specialWord() const;
  virtual uint32_t frameNumber() const;
  virtual uint32_t timeStampHi() const;
  virtual uint32_t timeStampLo() const;

  /** Frame data */
  ndarray<const uint16_t, 1> _data() const;

  virtual ndarray<const uint16_t, 2> data() const;

private:

  PNCCD::ns_FrameV1_v0::dataset_data m_ds_data;
  ndarray<const uint16_t, 1> m_ds_frameData;
};


template <typename Config>
class FramesV1_v0 : public Psana::PNCCD::FramesV1 {
public:
  typedef Psana::PNCCD::FramesV1 PsanaType;

  FramesV1_v0() {}
  FramesV1_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}

  virtual ~FramesV1_v0() {}

  /** Number of frames is determined by numLinks() method. */
  virtual const Psana::PNCCD::FrameV1& frame(uint32_t i0) const;

  virtual uint32_t numLinks() const { return m_cfg->numLinks(); }

  /** Method which returns the shape (dimensions) of the data returned by frame() method. */
  virtual std::vector<int> frame_shape() const;

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<Config> m_cfg;

  mutable ndarray<FrameV1_v0, 1> m_frames;
  void read_frames() const;
};


void make_datasets_FramesV1_v0(const Psana::PNCCD::FramesV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_FramesV1_v0(const Psana::PNCCD::FramesV1* obj, hdf5pp::Group group, long index, bool append);


class FullFrameV1_v0 : public Psana::PNCCD::FullFrameV1 {
public:
  typedef Psana::PNCCD::FramesV1 PsanaType;

  FullFrameV1_v0() {}
  FullFrameV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}

  virtual ~FullFrameV1_v0() {}

  /** Special values */
  virtual uint32_t specialWord() const;
  /** Frame number */
  virtual uint32_t frameNumber() const;
  /** Most significant part of timestamp */
  virtual uint32_t timeStampHi() const;
  /** Least significant part of timestamp */
  virtual uint32_t timeStampLo() const;
  /** Full frame data, image size is 1024x1024. */
  virtual ndarray<const uint16_t, 2> data() const;

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;

  mutable uint32_t m_specialWord;
  mutable uint32_t m_frameNumber;
  mutable uint32_t m_timeStampHi;
  mutable uint32_t m_timeStampLo;
  mutable ndarray<uint16_t, 2> m_frame;

  void read_frame() const;
};

void make_datasets_FullFrameV1_v0(const Psana::PNCCD::FullFrameV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_FullFrameV1_v0(const Psana::PNCCD::FullFrameV1* obj, hdf5pp::Group group, long index, bool append);

} // namespace PNCCD
} // namespace psddl_hdf2psana
#endif // PSDDL_HDF2PSANA_PNCCD_H
