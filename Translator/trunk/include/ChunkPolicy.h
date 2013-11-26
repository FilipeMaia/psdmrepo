#ifndef TRANSLATOR_CHUNKPOLICY_H
#define TRANSLATOR_CHUNKPOLICY_H

#include <vector>

#include "psddl_hdf2psana/ChunkPolicy.h"

namespace Translator {

class ChunkManager;

/**
 *  @ingroup Translator
 *
 *  @brief Copy of default implementation on the chunk size policy, with dynamic updating
 *
 *  This class copies psddl_hdf2psana::DefaultChunkPolicy and extends it to update chunk
 *  parameters after construction.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */

class ChunkPolicy : public psddl_hdf2psana::ChunkPolicy {
public:

  ChunkPolicy(hsize_t chunkSizeTargetBytes,
              int chunkSizeTarget = 0,
              hsize_t maxChunkSizeBytes = 100*1024*1024,
              int minObjectsPerChunk = 50,
              int maxObjectsPerChunk = 2048,
              hsize_t minChunkCacheSize = 1024*1024,
              hsize_t maxChunkCacheSize = 10*1024*1024);

  // Destructor
  virtual ~ChunkPolicy () ;
  
  virtual int chunkSize(const hdf5pp::Type& dsType) const;
  virtual int chunkSize(const size_t typeSize) const;

  virtual int chunkCacheSize(const hdf5pp::Type& dsType) const;
  virtual int chunkCacheSize(const size_t typeSize) const;

  friend class ChunkManager;

protected:
  hsize_t chunkSizeTargetBytes() const { return  m_chunkSizeTargetBytes;}
  hsize_t maxChunkSizeBytes() const { return  m_maxChunkSizeBytes;}
  int chunkSizeInElements() const { return  m_chunkSizeTarget;}
  int minObjectsPerChunk() const { return  m_minObjectsPerChunk;}
  int maxObjectsPerChunk() const { return  m_maxObjectsPerChunk;}
  hsize_t minChunkCacheSize() const { return  m_minChunkCacheSize;}
  hsize_t maxChunkCacheSize() const { return  m_maxChunkCacheSize;}

  void chunkSizeTargetBytes(hsize_t val) { m_chunkSizeTargetBytes = val;}
  void maxChunkSizeBytes(hsize_t val) { m_maxChunkSizeBytes = val;}
  void chunkSizeInElements(int val);
  void minObjectsPerChunk(int val) { m_minObjectsPerChunk = val;}
  void maxObjectsPerChunk(int val) { m_maxObjectsPerChunk = val;}
  void minChunkCacheSize(hsize_t val) { m_minChunkCacheSize = val;}
  void maxChunkCacheSize(hsize_t val) { m_maxChunkCacheSize = val;}

  void clearStats();
  void getStats(const std::vector<int> * & chunkCacheSizes,
                const std::vector<int> * & chunkSizes,
                const std::vector<size_t> * & objSizes);
private:
  mutable std::vector<int> returnedChunkCacheSizes;
  mutable std::vector<int> returnedChunkSizes;
  mutable std::vector<size_t> objSizesDuringChunkCacheCalls;

  hsize_t m_chunkSizeTargetBytes;
  hsize_t m_maxChunkSizeBytes;
  int m_chunkSizeTarget;
  int m_minObjectsPerChunk;
  int m_maxObjectsPerChunk;
  size_t m_minChunkCacheSize;
  hsize_t m_maxChunkCacheSize;

  // Copy constructor and assignment are disabled by default
  ChunkPolicy ( const ChunkPolicy& ) ;
  ChunkPolicy& operator = ( const ChunkPolicy& ) ;

};

} // namespace Translator

#endif // TRANSLATOR_CHUNKPOLICY_H
