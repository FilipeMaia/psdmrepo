#ifndef PSDDL_HDF2PSANA_CSPAD2X2_H
#define PSDDL_HDF2PSANA_CSPAD2X2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class cspad2x2.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psddl_psana/cspad2x2.ddl.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "psddl_hdf2psana/ChunkPolicy.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psddl_hdf2psana {
namespace CsPad2x2 {

namespace ns_ElementV1_v0 {
struct dataset_element {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_element();
  dataset_element(const Psana::CsPad2x2::ElementV1& psanaobj);

  uint32_t virtual_channel;
  uint32_t lane;
  uint32_t tid;
  uint32_t acq_count;
  uint32_t op_code;
  uint32_t quad;
  uint32_t seq_count;
  uint32_t ticks;
  uint32_t fiducials;
  uint32_t frame_type;
  uint16_t sb_temp[4];

};
}


class ElementV1_v0 : public Psana::CsPad2x2::ElementV1 {
public:

  typedef Psana::CsPad2x2::ElementV1 PsanaType;

  ElementV1_v0() {}
  ElementV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}

  virtual ~ElementV1_v0() {}

  virtual uint32_t virtual_channel() const;
  virtual uint32_t lane() const;
  virtual uint32_t tid() const;
  virtual uint32_t acq_count() const;
  virtual uint32_t op_code() const;
  virtual uint32_t quad() const;
  virtual uint32_t seq_count() const;
  virtual uint32_t ticks() const;
  virtual uint32_t fiducials() const;
  virtual ndarray<const uint16_t, 1> sb_temp() const;
  virtual uint32_t frame_type() const;
  virtual ndarray<const int16_t, 3> data() const;
  /** Common mode value for a given section, section number can be 0 or 1.
                Will return 0 for data read from XTC, may be non-zero after calibration. */
  float common_mode(uint32_t section) const;

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;

  mutable boost::shared_ptr<CsPad2x2::ns_ElementV1_v0::dataset_element> m_ds_element;
  void read_ds_element() const;
  
  mutable ndarray<const int16_t, 3> m_ds_data;
  void read_ds_data() const;

  mutable ndarray<float, 1> m_ds_cm;
  void read_ds_cm() const;
};

void make_datasets_ElementV1_v0(const Psana::CsPad2x2::ElementV1& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_ElementV1_v0(const Psana::CsPad2x2::ElementV1* obj, hdf5pp::Group group, long index, bool append);

} // namespace CsPad2x2
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_CSPAD2X2_H
