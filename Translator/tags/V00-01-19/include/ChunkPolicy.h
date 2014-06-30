#ifndef TRANSLATOR_CHUNKPOLICY_H
#define TRANSLATOR_CHUNKPOLICY_H

#include <vector>

#include "psddl_hdf2psana/ChunkPolicy.h"

namespace Translator {

class ChunkManager;

/**
 *  @ingroup Translator
 *
 *  @brief Copy of default implementation on the chunk size policy, with dynamic updating.
 *
 *  This class copies psddl_hdf2psana::DefaultChunkPolicy and extends it to update chunk
 *  parameters after construction. It deviates from DefaultChunkPolicy in how the 
 *  the per dataset chunk cache is created.  It uses a target of cache size of 2.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class ChunkPolicy : public psddl_hdf2psana::ChunkPolicy {
public:
  ChunkPolicy(hsize_t chunkSizeTargetBytes = 16*1024*1024,  // 16MB
              int chunkSizeTargetObjects = 0,
              hsize_t maxChunkSizeBytes = 100*1024*1024,    // 100MB
              int minObjectsPerChunk = 50,
              int maxObjectsPerChunk = 2048,
              int chunkCacheSizeTargetInChunks = 2,
              hsize_t maxChunkCacheSizeBytes = 100*1024*1024);  // 100MB

  // Destructor
  virtual ~ChunkPolicy () ;
  
  // returns chunk size in elements
  virtual int chunkSize(const hdf5pp::Type& dsType) const;
  virtual int chunkSize(const size_t typeSize) const;

  // returns chunk cache size in chunks
  virtual int chunkCacheSize(const hdf5pp::Type& dsType) const;
  virtual int chunkCacheSize(const size_t typeSize) const;

  friend class ChunkManager;

protected:
  hsize_t chunkSizeTargetBytes() const { return  m_chunkSizeTargetBytes;}
  int chunkSizeTargetObjects() const { return  m_chunkSizeTargetObjects;}
  hsize_t maxChunkSizeBytes() const { return  m_maxChunkSizeBytes;}
  int minObjectsPerChunk() const { return  m_minObjectsPerChunk;}
  int maxObjectsPerChunk() const { return  m_maxObjectsPerChunk;}
  hsize_t chunkCacheSizeTargetInChunks() const { return  m_chunkCacheSizeTargetInChunks;}
  hsize_t maxChunkCacheSizeBytes() const { return  m_maxChunkCacheSizeBytes;}

  void chunkSizeTargetBytes(hsize_t val) { m_chunkSizeTargetBytes = val;}
  void chunkSizeTargetObjects(int val);
  void maxChunkSizeBytes(hsize_t val) { m_maxChunkSizeBytes = val;}
  void minObjectsPerChunk(int val) { m_minObjectsPerChunk = val;}
  void maxObjectsPerChunk(int val) { m_maxObjectsPerChunk = val;}
  void chunkCacheSizeTargetInChunks(hsize_t val) {m_chunkCacheSizeTargetInChunks = val;}
  void maxChunkCacheSizeBytes(hsize_t val) { m_maxChunkCacheSizeBytes = val;}

private:
  hsize_t m_chunkSizeTargetBytes;
  int  m_chunkSizeTargetObjects;
  hsize_t  m_maxChunkSizeBytes;
  int  m_minObjectsPerChunk;
  int  m_maxObjectsPerChunk;
  int  m_chunkCacheSizeTargetInChunks;
  hsize_t  m_maxChunkCacheSizeBytes;

  // Copy constructor and assignment are disabled by default
  ChunkPolicy ( const ChunkPolicy& ) ;
  ChunkPolicy& operator = ( const ChunkPolicy& ) ;

};

} // namespace Translator

#endif // TRANSLATOR_CHUNKPOLICY_H
