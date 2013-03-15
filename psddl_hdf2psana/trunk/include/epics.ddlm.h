#ifndef PSDDL_HDF2PSANA_EPICS_DDLM_H
#define PSDDL_HDF2PSANA_EPICS_DDLM_H 1

#include "psddl_psana/epics.ddl.h"

#include "hdf5pp/Group.h"
#include "hdf5pp/Type.h"
#include "PSEnv/EpicsStore.h"

namespace psddl_hdf2psana {
namespace Epics {

  /**
   *  Read data from specified group and convert them into EPICS object
   */
  boost::shared_ptr<Psana::Epics::EpicsPvHeader> readEpics(const hdf5pp::Group& group, int64_t idx);


} // namespace Epics
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_EPICS_DDLM_H
