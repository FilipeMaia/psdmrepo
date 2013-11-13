#ifndef PSDDL_HDF2PSANA_EPICS_H
#define PSDDL_HDF2PSANA_EPICS_H 1

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

#endif // PSDDL_HDF2PSANA_EPICS_H
