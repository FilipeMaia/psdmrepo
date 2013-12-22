#ifndef PSDDL_HDF2PSANA_CHUNKPOLICY_H
#define PSDDL_HDF2PSANA_CHUNKPOLICY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ChunkPolicy.
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
 *  @brief Interface for policy instances which define chunk size of datasets.
 *
 *  The code which is going to create datasets by calling one of many
 *  make_dataset() methods will have to create instance of the policy class
 *  and pass it to the method. There will be pre-defined policies but clients
 *  may chose to provide different implementation of necessary.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class ChunkPolicy  {
public:

  // Destructor
  virtual ~ChunkPolicy () ;

  /**
   *  @brief Return chunk size in objects for a dataset.
   *
   *  To help policy to decide what is the best chunk size we pass
   *  the type of data stored in the dataset.
   *
   *  @param[in] dsType   data type stored in a dataset.
   *  @return Size of chunk in objects.
   */
  virtual int chunkSize(const hdf5pp::Type& dsType) const = 0;

  /**
   *  @brief Return chunk cache size (in chunks) for a dataset.
   *
   *  Returns optimal size of chunk cache based on the type of a dataset.
   *  Size is returned as a number of chunks in a cache.
   *
   *  @param[in]  dsType       Data type of the stored data
   *  @return Size cache in units of chunks.
   */
  virtual int chunkCacheSize(const hdf5pp::Type& dsType) const = 0;

protected:

  // Default constructor
  ChunkPolicy () {}

private:

  // Copy constructor and assignment are disabled by default
  ChunkPolicy ( const ChunkPolicy& ) ;
  ChunkPolicy& operator = ( const ChunkPolicy& ) ;

};

} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_CHUNKPOLICY_H
