#ifndef PSDDL_HDF2PSANA_DEFAULTCHUNKPOLICY_H
#define PSDDL_HDF2PSANA_DEFAULTCHUNKPOLICY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DefaultChunkPolicy.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psddl_hdf2psana/ChunkPolicy.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

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
 *  @brief Default implementation on the chunk size policy.
 *
 *  This class provides default standard algorithm for calculating chunk size
 *  which is based on the code used by old translator. It has a bunch of
 *  parameters that changed changed by client via constructor parameters.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class DefaultChunkPolicy : public ChunkPolicy {
public:

  /**
   *  @brief Constructor takes a number of parameters needed by algorithm.
   *
   *  Clients must specify target size of chunks in bytes. Additionally if there is
   *  an estimate of how many objects the chunk is going to have (for example from
   *  parameters of calib cycle) it can be provided as a second argument, in this
   *  case first argument is ignored and target size in bytes will be determined
   *  from second argument (by multiplying second argument by HDF5 type size).
   *  Additional optional arguments define other parameters of the algorithm, the
   *  chunk size in object will be kept in range between minObjectsPerChunk and
   *  maxObjectsPerChunk, but the resulting chunk size will never be larger than
   *  maxChunkSizeBytes (unless single object is larger than that size).
   *
   *  Other parameters determine the size of the chunk cache, algorithm tries to
   *  select optimal size between minChunkCacheSize and maxChunkCacheSize but makes
   *  sure that at least one chunk fits in cache in any case.
   *
   *  @param[in] chunkSizeTargetBytes  Target size of chunks in bytes.
   *  @param[in] chunkSizeTarget       Target size of chunks in objects, if non zero
   *                                   overrides chunkSizeTargetBytes.
   *  @param[in] maxChunkSizeBytes     Absolute upper limit on chunk size in bytes.
   *  @param[in] maxObjectsPerChunk    Maximum number of objects in chunk.
   *  @param[in] minObjectsPerChunk    Minimum number of objects in chunk.
   *  @param[in] minChunkCacheSize     Minimum size of a chunk cache.
   *  @param[in] maxChunkCacheSize     Maximum size of a chunk cache.
   */
  DefaultChunkPolicy(hsize_t chunkSizeTargetBytes,
      int chunkSizeTarget = 0,
      hsize_t maxChunkSizeBytes = 100*1024*1024,
      int minObjectsPerChunk = 50,
      int maxObjectsPerChunk = 2048,
      hsize_t minChunkCacheSize = 1024*1024,
      hsize_t maxChunkCacheSize = 10*1024*1024);

  // Destructor
  virtual ~DefaultChunkPolicy () ;

  /**
   *  @brief Return chunk size in objects for a dataset
   *
   *  To help policy to decide what is the best chunk size we pass
   *  the type of data stored in the dataset.
   *
   *  @param[in] dsType   data type stored in a dataset.
   */
  virtual int chunkSize(const hdf5pp::Type& dsType) const;

  /**
   *  @brief Return chunk cache size (in chunks) for a dataset.
   *
   *  Returns optimal size of chunk cache based on the type of a dataset.
   *  Size is returned as a number of chunks in a cache.
   *
   *  @param[in]  dsType       Data type of the stored data
   *  @return Size cache in units of chunks.
   */
  virtual int chunkCacheSize(const hdf5pp::Type& dsType) const;

protected:

private:

  const hsize_t m_chunkSizeTargetBytes;
  const hsize_t m_maxChunkSizeBytes;
  const int m_chunkSizeTarget;
  const int m_minObjectsPerChunk;
  const int m_maxObjectsPerChunk;
  const hsize_t m_minChunkCacheSize;
  const hsize_t m_maxChunkCacheSize;

  // Copy constructor and assignment are disabled by default
  DefaultChunkPolicy ( const DefaultChunkPolicy& ) ;
  DefaultChunkPolicy& operator = ( const DefaultChunkPolicy& ) ;

};

} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_DEFAULTCHUNKPOLICY_H
