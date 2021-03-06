#ifndef PSDDL_HDF2PSANA_FCCD_DDL_H
#define PSDDL_HDF2PSANA_FCCD_DDL_H 1

// *** Do not edit this file, it is auto-generated ***

#include "psddl_psana/fccd.ddl.h"
#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "PSEvt/Proxy.h"
#include "psddl_hdf2psana/fli.ddl.h"
namespace psddl_hdf2psana {
namespace FCCD {

namespace ns_FccdConfigV1_v0 {
struct dataset_config {
  static hdf5pp::Type native_type();
  static hdf5pp::Type stored_type();

  dataset_config();
  ~dataset_config();

  uint16_t outputMode; 
  uint32_t width; 
  uint32_t height; 
  uint32_t trimmedWidth; 
  uint32_t trimmedHeight; 

};
}


class FccdConfigV1_v0 : public Psana::FCCD::FccdConfigV1 {
public:
  typedef Psana::FCCD::FccdConfigV1 PsanaType;
  FccdConfigV1_v0() {}
  FccdConfigV1_v0(hdf5pp::Group group, hsize_t idx)
    : m_group(group), m_idx(idx) {}
  FccdConfigV1_v0(const boost::shared_ptr<FCCD::ns_FccdConfigV1_v0::dataset_config>& ds) : m_ds_config(ds) {}
  virtual ~FccdConfigV1_v0() {}
  virtual uint16_t outputMode() const;
  virtual uint32_t width() const;
  virtual uint32_t height() const;
  virtual uint32_t trimmedWidth() const;
  virtual uint32_t trimmedHeight() const;
private:
  mutable hdf5pp::Group m_group;
  hsize_t m_idx;
  mutable boost::shared_ptr<FCCD::ns_FccdConfigV1_v0::dataset_config> m_ds_config;
  void read_ds_config() const;
};

boost::shared_ptr<PSEvt::Proxy<Psana::FCCD::FccdConfigV1> > make_FccdConfigV1(int version, hdf5pp::Group group, hsize_t idx);
boost::shared_ptr<PSEvt::Proxy<Psana::FCCD::FccdConfigV2> > make_FccdConfigV2(int version, hdf5pp::Group group, hsize_t idx);
} // namespace FCCD
} // namespace psddl_hdf2psana
#endif // PSDDL_HDF2PSANA_FCCD_DDL_H
