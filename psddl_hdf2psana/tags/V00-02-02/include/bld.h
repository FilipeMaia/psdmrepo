#ifndef PSDDL_HDF2PSANA_BLD_H
#define PSDDL_HDF2PSANA_BLD_H

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
#include "psddl_hdf2psana/camera.h"
#include "psddl_hdf2psana/lusi.ddl.h"
#include "psddl_hdf2psana/pulnix.ddl.h"
#include "psddl_psana/bld.ddl.h"

namespace psddl_hdf2psana {
namespace Bld {

namespace ns_BldDataPimV1_v0 {
struct dataset_data {
  static hdf5pp::Type native_type() ;
  static hdf5pp::Type stored_type() ;

  Pulnix::ns_TM6740ConfigV2_v0::dataset_config camConfig;
  Lusi::ns_PimImageConfigV1_v0::dataset_config pimConfig;
  Camera::ns_FrameV1_v0::dataset_data frame;

};
}

class BldDataPimV1_v0 : public Psana::Bld::BldDataPimV1 {
public:

  typedef Psana::Bld::BldDataPimV1 PsanaType;

  BldDataPimV1_v0() {}
  BldDataPimV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}

  virtual ~BldDataPimV1_v0() {}

  virtual const Psana::Pulnix::TM6740ConfigV2& camConfig() const;
  virtual const Psana::Lusi::PimImageConfigV1& pimConfig() const;
  virtual const Psana::Camera::FrameV1& frame() const;

private:

  mutable hdf5pp::Group m_group;
  hsize_t m_idx;

  mutable boost::shared_ptr<Bld::ns_BldDataPimV1_v0::dataset_data> m_ds_data;
  void read_ds_data() const;

  mutable boost::shared_ptr<Psana::Pulnix::TM6740ConfigV2> m_storage_camConfig;
  mutable Psana::Lusi::PimImageConfigV1 m_storage_pimConfig;
  mutable boost::shared_ptr<Psana::Camera::FrameV1> m_storage_frame;

};

void store_BldDataPimV1_v0(const Psana::Bld::BldDataPimV1& obj, hdf5pp::Group group, bool append);

} // namespace Bld
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_BLD.DDLM_H
