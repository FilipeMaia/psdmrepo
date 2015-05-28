#ifndef PSDDL_HDF2PSANA_CSPAD_H
#define PSDDL_HDF2PSANA_CSPAD_H 1

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
#include "psddl_hdf2psana/cspad.ddl.h"
#include "psddl_hdf2psana/ChunkPolicy.h"


namespace psddl_hdf2psana {
namespace CsPad {


class DataV1_v0_Element : public Psana::CsPad::ElementV1 {
public:
  typedef Psana::CsPad::ElementV1 PsanaType;

  DataV1_v0_Element() {}
  DataV1_v0_Element(const CsPad::ns_ElementV1_v0::dataset_element& ds_elem,
      ndarray<int16_t, 3> ds_data, ndarray<float, 1> ds_cm)
    : m_ds_elem(ds_elem), m_ds_data(ds_data), m_ds_cm(ds_cm) {}

  virtual ~DataV1_v0_Element() {}

  virtual uint32_t virtual_channel() const { return m_ds_elem.virtual_channel; }
  virtual uint32_t lane() const { return m_ds_elem.lane; }
  virtual uint32_t tid() const { return m_ds_elem.tid; }
  virtual uint32_t acq_count() const { return m_ds_elem.acq_count; }
  virtual uint32_t op_code() const { return m_ds_elem.op_code; }
  virtual uint32_t quad() const { return m_ds_elem.quad; }
  virtual uint32_t seq_count() const { return m_ds_elem.seq_count; }
  virtual uint32_t ticks() const { return m_ds_elem.ticks; }
  virtual uint32_t fiducials() const { return m_ds_elem.fiducials; }
  virtual ndarray<const uint16_t, 1> sb_temp() const {
    ndarray<uint16_t, 1> res = make_ndarray<uint16_t>(4);
    std::copy(m_ds_elem.sb_temp, m_ds_elem.sb_temp+4, res.begin());
    return res;
  }
  virtual uint32_t frame_type() const { return m_ds_elem.frame_type; }
  virtual ndarray<const int16_t, 3> data() const { return m_ds_data; }
  virtual uint32_t sectionMask() const { return m_ds_elem.sectionMask; }

  /** Common mode value for a given section, section number can be 0 to config.numAsicsRead()/2.
                Will return 0 for data read from XTC, may be non-zero after calibration. */
  float common_mode(uint32_t section) const { return m_ds_cm[section]; }

private:

  CsPad::ns_ElementV1_v0::dataset_element m_ds_elem;
  ndarray<const int16_t, 3> m_ds_data;
  ndarray<float, 1> m_ds_cm;
};


template <typename Config>
class DataV1_v0 : public Psana::CsPad::DataV1 {
public:
  typedef Psana::CsPad::DataV1 PsanaType;
  DataV1_v0() {}
  DataV1_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}
  virtual ~DataV1_v0() {}

  /** Data objects, one element per quadrant. The size of the array is determined by
            the numQuads() method of the configuration object. */
  virtual const Psana::CsPad::ElementV1& quads(uint32_t i0) const;
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  virtual std::vector<int> quads_shape() const;

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<Config> m_cfg;

  mutable ndarray<DataV1_v0_Element, 1> m_elements;
  void read_elements() const;
};



void make_datasets_DataV1_v0(const Psana::CsPad::DataV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_DataV1_v0(const Psana::CsPad::DataV1* obj, hdf5pp::Group group, long index, bool append);




class DataV2_v0_Element : public Psana::CsPad::ElementV2 {
public:
  typedef Psana::CsPad::ElementV2 PsanaType;

  DataV2_v0_Element() {}
  DataV2_v0_Element(const CsPad::ns_ElementV2_v0::dataset_element& ds_elem,
      ndarray<int16_t, 3> ds_data, ndarray<float, 1> ds_cm)
    : m_ds_elem(ds_elem), m_ds_data(ds_data), m_ds_cm(ds_cm) {}

  virtual ~DataV2_v0_Element() {}

  virtual uint32_t virtual_channel() const { return m_ds_elem.virtual_channel; }
  virtual uint32_t lane() const { return m_ds_elem.lane; }
  virtual uint32_t tid() const { return m_ds_elem.tid; }
  virtual uint32_t acq_count() const { return m_ds_elem.acq_count; }
  virtual uint32_t op_code() const { return m_ds_elem.op_code; }
  virtual uint32_t quad() const { return m_ds_elem.quad; }
  virtual uint32_t seq_count() const { return m_ds_elem.seq_count; }
  virtual uint32_t ticks() const { return m_ds_elem.ticks; }
  virtual uint32_t fiducials() const { return m_ds_elem.fiducials; }
  virtual ndarray<const uint16_t, 1> sb_temp() const {
    ndarray<uint16_t, 1> res = make_ndarray<uint16_t>(4);
    std::copy(m_ds_elem.sb_temp, m_ds_elem.sb_temp+4, res.begin());
    return res;
  }
  virtual uint32_t frame_type() const { return m_ds_elem.frame_type; }
  virtual ndarray<const int16_t, 3> data() const { return m_ds_data; }
  virtual uint32_t sectionMask() const { return m_ds_elem.sectionMask; }

  /** Common mode value for a given section, section number can be 0 to config.numSect().
                Will return 0 for data read from XTC, may be non-zero after calibration. */
  float common_mode(uint32_t section) const { return m_ds_cm[section]; }

private:

  CsPad::ns_ElementV2_v0::dataset_element m_ds_elem;
  ndarray<const int16_t, 3> m_ds_data;
  ndarray<float, 1> m_ds_cm;
};




template <typename Config>
class DataV2_v0 : public Psana::CsPad::DataV2 {
public:
  typedef Psana::CsPad::DataV1 PsanaType;
  DataV2_v0() {}
  DataV2_v0(hdf5pp::Group group, hsize_t idx, const boost::shared_ptr<Config>& cfg)
    : m_group(group), m_idx(idx), m_cfg(cfg) {}
  virtual ~DataV2_v0() {}

  /** Data objects, one element per quadrant. The size of the array is determined by
            the numQuads() method of the configuration object. */
  virtual const Psana::CsPad::ElementV2& quads(uint32_t i0) const;
  /** Method which returns the shape (dimensions) of the data returned by quads() method. */
  virtual std::vector<int> quads_shape() const;

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  boost::shared_ptr<Config> m_cfg;

  mutable ndarray<DataV2_v0_Element, 1> m_elements;
  void read_elements() const;
};

void make_datasets_DataV2_v0(const Psana::CsPad::DataV2& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_DataV2_v0(const Psana::CsPad::DataV2* obj, hdf5pp::Group group, long index, bool append);

} // namespace CsPad
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_CSPAD_H
