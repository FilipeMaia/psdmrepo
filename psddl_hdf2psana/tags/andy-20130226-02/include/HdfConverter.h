#ifndef PSDDL_HDF2PSANA_HDFCONVERTER_H
#define PSDDL_HDF2PSANA_HDFCONVERTER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class HdfConverter.
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
#include "pdsdata/xtc/Src.hh"
#include "pdsdata/xtc/TypeId.hh"
#include "PSEnv/EnvObjectStore.h"
#include "PSEnv/EpicsStore.h"
#include "PSEvt/Event.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psddl_hdf2psana {

/// @addtogroup psddl_hdf2psana

/**
 *  @ingroup psddl_hdf2psana
 *
 *  @brief Class which implements conversion of HDF5 data into psana objects
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class HdfConverter : boost::noncopyable {
public:

  // Default constructor
  HdfConverter () ;

  // Destructor
  ~HdfConverter () ;

  /**
   *  @brief Convert one object and store it in the event.
   */
  void convert(const hdf5pp::Group& group, uint64_t idx, PSEvt::Event& evt, PSEnv::EnvObjectStore& cfgStore);
  
  /**
   *  @brief Convert one object and store it in the config store.
   */
  void convertConfig(const hdf5pp::Group& group, uint64_t idx, PSEnv::EnvObjectStore& cfgStore);

  /**
   *  @brief Convert one object and store it in the epics store.
   */
  void convertEpics(const hdf5pp::Group& group, uint64_t idx, PSEnv::EpicsStore& eStore);

  /**
   *  @brief This method should be called to reset cache whenever some groups are closed
   */
  void resetCache();

protected:

  // test if the group is or inside EPICS group (named Epics::EpicsPv),
  // check only limited number of levels up
  bool isEpics(const hdf5pp::Group& group, int levels=2) const;

  // Get schema version for the group or its parent (and its grand-parent for EPICS),
  int schemaVersion(const hdf5pp::Group& group, int levels = -1) const;

  // Get TypeId for the group or its parent (and its grand-parent for EPICS),
  Pds::TypeId typeId(const hdf5pp::Group& group, int levels = -1) const;

  // Get Source for the group (or its parent for EPICS),
  Pds::Src source(const hdf5pp::Group& group, int levels = -1) const;

private:

  mutable std::map<hdf5pp::Group, int> m_schemaVersionCache;
  mutable std::map<hdf5pp::Group, bool> m_isEpicsCache;
  mutable std::map<hdf5pp::Group, Pds::TypeId> m_typeIdCache;
  mutable std::map<hdf5pp::Group, Pds::Src> m_sourceCache;

};

} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_HDFCONVERTER_H
