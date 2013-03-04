#ifndef PSDDL_HDF2PSANA_CAMERA_DDLM_H
#define PSDDL_HDF2PSANA_CAMERA_DDLM_H

#include <boost/shared_ptr.hpp>

#include "hdf5pp/Group.h"
#include "psddl_psana/camera.ddl.h"

namespace psddl_hdf2psana {
namespace Camera {

namespace ns_FrameV1_v0 {
struct dataset_image {
  static hdf5pp::Type stored_type();
  static hdf5pp::Type native_type();
  size_t vlen_data;
  uint16_t* data;
}; // class dataset_image
} // namespace ns_FrameV1_v0

} // namespace Camera
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_CAMERA_DDLM_H
