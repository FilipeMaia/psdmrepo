//-----------------------
// This Class's Header --
//-----------------------
#include "Translator/ChunkPolicy.h"
#include "MsgLogger/MsgLogger.h"
//-----------------
// C/C++ Headers --
//-----------------
//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace {
  const char * logger = "Translator.ChunkPolicy";
}

namespace Translator {

//----------------
// Constructors --
//----------------
ChunkPolicy::ChunkPolicy(hsize_t chunkSizeTargetBytes,
    int chunkSizeTarget,
    hsize_t maxChunkSizeBytes,
    int minObjectsPerChunk,
    int maxObjectsPerChunk,
    hsize_t minChunkCacheSize,
    hsize_t maxChunkCacheSize)
  : m_chunkSizeTargetBytes(chunkSizeTargetBytes)
  , m_maxChunkSizeBytes(maxChunkSizeBytes)
  , m_chunkSizeTarget(chunkSizeTarget)
  , m_minObjectsPerChunk(minObjectsPerChunk)
  , m_maxObjectsPerChunk(maxObjectsPerChunk)
  , m_minChunkCacheSize(minChunkCacheSize)
  , m_maxChunkCacheSize(maxChunkCacheSize)
{
  MsgLog(logger,info,"chunkSizeTarget = " << m_chunkSizeTarget);
}

//--------------
// Destructor --
//--------------
ChunkPolicy::~ChunkPolicy ()
{
}

// Return chunk size in objects for a dataset
int ChunkPolicy::chunkSize(const hdf5pp::Type& dsType) const {
  const size_t obj_size = dsType.size();
  return chunkSize(obj_size);
}

int ChunkPolicy::chunkSize(size_t obj_size) const {
  int objectsPerChunk = m_chunkSizeTarget > 0 ? m_chunkSizeTarget : m_chunkSizeTargetBytes / obj_size;
  objectsPerChunk = std::min(objectsPerChunk, m_maxObjectsPerChunk);
  objectsPerChunk = std::max(objectsPerChunk, m_minObjectsPerChunk);
  int maxObjectsThatFitInMaxChunkSizeBytes = std::max((hsize_t)1,m_maxChunkSizeBytes/obj_size);
  objectsPerChunk = std::min(objectsPerChunk, maxObjectsThatFitInMaxChunkSizeBytes);

  MsgLog(logger,debug,"Translator::ChunkPolicy chunkSize= " << objectsPerChunk
         << " typeSize=" << obj_size << " chunkSizeTarget=" << m_chunkSizeTarget 
         << " chunkSizeTargetBytes=" << m_chunkSizeTargetBytes 
         << " chunkSizeBounds=[" << m_minObjectsPerChunk << ", " << m_maxObjectsPerChunk << "]"
         << " maxChunkBytes=" << m_maxChunkSizeBytes 
         << " most objects we can fit in maxChunkBytes=" << maxObjectsThatFitInMaxChunkSizeBytes);

  returnedChunkSizes.push_back(objectsPerChunk);

  return objectsPerChunk;
}

// Return chunk cache size (in chunks) for a dataset.
int ChunkPolicy::chunkCacheSize(const hdf5pp::Type& dsType) const
{
  const size_t obj_size = dsType.size();
  return chunkCacheSize(obj_size);
}

int ChunkPolicy::chunkCacheSize(const size_t obj_size) const {
  int cacheSizeInChunks = 30;
  MsgLog(logger,debug,"Translator::ChunkPolicy::chunkCacheSize: " << cacheSizeInChunks);
  return cacheSizeInChunks;
  /*
  const int chunk_size = chunkSize(obj_size);
  hsize_t chunk_size_bytes = chunk_size * obj_size;
  hsize_t chunk_cache_size = 1;
  if (chunk_size_bytes <= m_minChunkCacheSize/2) {
    chunk_cache_size = m_minChunkCacheSize/chunk_size_bytes;
  } else if (chunk_size_bytes <= m_maxChunkCacheSize/2) {
    chunk_cache_size *= 2;
  }

  objSizesDuringChunkCacheCalls.push_back(obj_size);
  returnedChunkCacheSizes.push_back(chunk_cache_size);
  return chunk_cache_size;
  */
}

void ChunkPolicy::clearStats() {
  returnedChunkCacheSizes.clear();
  returnedChunkSizes.clear();
  objSizesDuringChunkCacheCalls.clear();
}

void ChunkPolicy::getStats(const std::vector<int> * &chunkCacheSizes,
                           const std::vector<int> * &chunkSizes,
                           const std::vector<size_t> * &objSizes) {
  MsgLog(logger,debug,"cache = " << (void *)&returnedChunkCacheSizes
         << " chunk = " << (void *)&returnedChunkSizes
         << " obj = " << (void *)&objSizesDuringChunkCacheCalls);

  MsgLog(logger,debug,"cache = " << (void *)chunkCacheSizes
         << " chunk = " << (void *)chunkSizes
         << " obj = " << (void *)objSizes);

  chunkCacheSizes = &returnedChunkCacheSizes;
  chunkSizes      = &returnedChunkSizes;
  objSizes        = &objSizesDuringChunkCacheCalls;

  MsgLog(logger,debug,"cache = " << (void *)chunkCacheSizes
         << " chunk = " << (void *)chunkSizes
         << " obj = " << (void *)objSizes);
}

void ChunkPolicy::chunkSizeInElements(int val) {
  MsgLog(logger,info,"chunkSizeInElementsTarget = " << val);
  m_chunkSizeTarget = val;
}

} // namespace Translator
