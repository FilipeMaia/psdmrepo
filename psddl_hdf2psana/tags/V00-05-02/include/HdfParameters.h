#ifndef PSDDL_HDF2PSANA_HDFPARAMETERS_H
#define PSDDL_HDF2PSANA_HDFPARAMETERS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class HdfParameters.
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
#include "hdf5pp/Type.h"

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
 *  @brief Utility class that parameterizes some aspect of reading/writing HDF5.
 *
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class HdfParameters  {
public:

  /**
   *  @brief Method that calculates chunk cache size for given storage type.
   *
   *  Returns optimal size of chunk cache based on the size of the provided type.
   *  Size is returned as a number of chunks in a cache.
   *
   *  @param[in]  type       Data type of the stored data
   *  @param[in]  chunk_size Size of single chunk in units of type.
   *  @return Size cache in units of chunks.
   */
  static unsigned chunkCacheSize(const hdf5pp::Type& type, unsigned chunk_size);

private:

};

} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_HDFPARAMETERS_H
