#ifndef PSDDL_HDF2PSANA_ACQIRIS_H
#define PSDDL_HDF2PSANA_ACQIRIS_H

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
#include "psddl_psana/acqiris.ddl.h"
#include "psddl_hdf2psana/acqiris.ddl.h"

namespace psddl_hdf2psana {
namespace Acqiris {


template <typename Config>
class DataDescV1Elem_v0 : public Psana::Acqiris::DataDescV1Elem {
public:
  typedef Psana::Acqiris::DataDescV1Elem PsanaType;

  DataDescV1Elem_v0() {}
  // this is not useful but needs to be defined
  DataDescV1Elem_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_cfg(cfg) {}
  DataDescV1Elem_v0(const ndarray<const Psana::Acqiris::TimestampV1, 1>& ds_timestamp,
      const ndarray<const int16_t, 2>& ds_waveforms,
      const boost::shared_ptr<Config>& cfg)
    : m_cfg(cfg), m_ds_timestamp(ds_timestamp), m_ds_waveforms(ds_waveforms) {}

  virtual ~DataDescV1Elem_v0() {}

  virtual uint32_t nbrSamplesInSeg() const { return m_cfg->horiz().nbrSamples(); }
  virtual uint32_t indexFirstPoint() const { return 0; }
  virtual uint32_t nbrSegments() const { return m_cfg->horiz().nbrSegments(); }
  virtual ndarray<const Psana::Acqiris::TimestampV1, 1> timestamp() const { return m_ds_timestamp; }
  virtual ndarray<const int16_t, 2> waveforms() const { return m_ds_waveforms; }

private:

  boost::shared_ptr<Config> m_cfg;
  mutable ndarray<const Psana::Acqiris::TimestampV1, 1> m_ds_timestamp;
  mutable ndarray<const int16_t, 2> m_ds_waveforms;
};



template <typename Config>
class DataDescV1_v0 : public Psana::Acqiris::DataDescV1 {
public:
  typedef Psana::Acqiris::DataDescV1 PsanaType;

  DataDescV1_v0() {}
  DataDescV1_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}

  virtual ~DataDescV1_v0() {}

  /** Waveform data, one object per channel. */
  virtual const Psana::Acqiris::DataDescV1Elem& data(uint32_t i0) const;
  /** Method which returns the shape (dimensions) of the data returned by data() method. */
  virtual std::vector<int> data_shape() const;

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<Config> m_cfg;

  mutable ndarray<DataDescV1Elem_v0<Config>, 1> m_ds_data;
  void read_ds_data() const;

};


} // namespace Acqiris
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_ACQIRIS_H
