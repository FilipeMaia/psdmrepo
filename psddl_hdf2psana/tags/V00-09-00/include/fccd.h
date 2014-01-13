#ifndef PSDDL_HDF2PSANA_FCCD_H
#define PSDDL_HDF2PSANA_FCCD_H 1

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
#include "PSEvt/Proxy.h"
#include "psddl_hdf2psana/ChunkPolicy.h"
#include "psddl_psana/fccd.ddl.h"

namespace psddl_hdf2psana {
namespace FCCD {


namespace ns_FccdConfigV2_v0 {
struct dataset_config {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_config();
  dataset_config(const Psana::FCCD::FccdConfigV2& psanaobj);
  ~dataset_config();

  uint16_t outputMode; 
  uint8_t ccdEnable; 
  uint8_t focusMode; 
  uint32_t exposureTime; 
  uint32_t width; 
  uint32_t height; 
  uint32_t trimmedWidth; 
  uint32_t trimmedHeight; 
  float dacVoltage[Psana::FCCD::FccdConfigV2::NVoltages];
  uint16_t waveform[Psana::FCCD::FccdConfigV2::NWaveforms];
};
}


class FccdConfigV2_v0 : public Psana::FCCD::FccdConfigV2 {
public:
  typedef Psana::FCCD::FccdConfigV2 PsanaType;
  FccdConfigV2_v0() {}
  FccdConfigV2_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  virtual ~FccdConfigV2_v0() {}
  virtual uint16_t outputMode() const;
  virtual uint8_t ccdEnable() const;
  virtual uint8_t focusMode() const;
  virtual uint32_t exposureTime() const;
  virtual ndarray<const float, 1> dacVoltages() const;
  virtual ndarray<const uint16_t, 1> waveforms() const;
  virtual uint32_t width() const;
  virtual uint32_t height() const;
  virtual uint32_t trimmedWidth() const;
  virtual uint32_t trimmedHeight() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<FCCD::ns_FccdConfigV2_v0::dataset_config> m_ds_config;
  void read_ds_config() const;
};

void make_datasets_FccdConfigV2_v0(const Psana::FCCD::FccdConfigV2& obj,
      hdf5pp::Group group, const ChunkPolicy& chunkPolicy, int deflate, bool shuffle);
void store_FccdConfigV2_v0(const Psana::FCCD::FccdConfigV2* obj, hdf5pp::Group group, long index, bool append);

} // namespace FCCD
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_FCCD_H
