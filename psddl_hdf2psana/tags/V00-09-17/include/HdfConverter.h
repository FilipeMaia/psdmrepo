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
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "hdf5pp/DataSet.h"
#include "pdsdata/xtc/Src.hh"
#include "pdsdata/xtc/TypeId.hh"
#include "PSEnv/Env.h"
#include "PSEnv/EpicsStore.h"
#include "PSEvt/Event.h"
#include "psddl_hdf2psana/NDArrayConverter.h"

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
  void convert(const hdf5pp::Group& group, int64_t idx, PSEvt::Event& evt, PSEnv::Env& env);
  
  /**
   *  @brief Convert one object and store it in the epics store.
   */
  void convertEpics(const hdf5pp::Group& group, int64_t idx, PSEnv::EpicsStore& eStore);

  /**
   *  @brief This method should be called to reset cache whenever some groups are closed
   */
  void resetCache();

protected:

  // Get schema version for the group or its parent (and its grand-parent for EPICS),
  int schemaVersion(const hdf5pp::Group& group, int levels = -1) const;

  // Get type name for the parent of this group (or its grand-parent for EPICS),
  std::string typeName(const std::string& group) const;

  // Get Source for the group (or its parent for EPICS),
  Pds::Src source(const hdf5pp::Group& group, int levels = -1) const;

  // return true if this the parent is an ndarray 
  bool isNDArray(const hdf5pp::Group& group) const;
private:

  mutable std::map<std::string, int> m_schemaVersionCache;
  mutable std::map<std::string, Pds::Src> m_sourceCache;

  NDArrayConverter m_ndarrayConverter;
};

} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_HDFCONVERTER_H
