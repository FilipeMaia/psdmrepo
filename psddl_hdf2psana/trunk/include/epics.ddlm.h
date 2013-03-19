#ifndef PSDDL_HDF2PSANA_EPICS_DDLM_H
#define PSDDL_HDF2PSANA_EPICS_DDLM_H 1

#include "psddl_psana/epics.ddl.h"

#include "hdf5pp/DataSet.h"
#include "psddl_psana/epics.ddl.h"

namespace psddl_hdf2psana {
namespace Epics {

  /**
   *  Read data from specified group and convert them into EPICS object
   */
  boost::shared_ptr<Psana::Epics::EpicsPvHeader> readEpics(const hdf5pp::DataSet& ds, int64_t idx);
  boost::shared_ptr<Psana::Epics::EpicsPvHeader> readEpics(const hdf5pp::DataSet& ds, int64_t idx, 
      const Psana::Epics::EpicsPvHeader& pvhdr);


} // namespace Epics
} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_EPICS_DDLM_H
