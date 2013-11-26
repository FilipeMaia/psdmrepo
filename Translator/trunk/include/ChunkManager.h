#ifndef TRANSLATOR_CHUNK_MANAGER_H
#define TRANSLATOR_CHUNK_MANAGER_H

#include <vector>

#include "PSEnv/Env.h"
#include "Translator/ChunkPolicy.h"

namespace Translator {

class H5Output;

class ChunkManager {
 public:
  ChunkManager();
  ~ChunkManager();
  void readConfigParameters(const Translator::H5Output &);
  void beginJob(PSEnv::Env &env);
  void beginCalibCycle(PSEnv::Env &env);
  void endCalibCycle(size_t numberEventsInCalibCycle);
  void endJob() {};

  boost::shared_ptr<Translator::ChunkPolicy> eventIdChunkPolicy() { return m_eventIdChunkPolicy; }
  boost::shared_ptr<Translator::ChunkPolicy> damageChunkPolicy() { return m_damageChunkPolicy; }
  boost::shared_ptr<Translator::ChunkPolicy> filterMsgChunkPolicy() { return m_filterMsgChunkPolicy; }
  boost::shared_ptr<Translator::ChunkPolicy> epicsPvChunkPolicy() { return m_epicsPvChunkPolicy; }
  boost::shared_ptr<Translator::ChunkPolicy> defaultChunkPolicy() { return m_defaultChunkPolicy; }

 protected:
  void checkForControlDataToSetChunkSize(PSEnv::Env &env);
  void clearStats();
  void reportStats();
  void setChunkSizeInElements(int chunkSizeInElements);

 private:
  hsize_t m_chunkSizeInBytes;
  int m_chunkSizeInElements, m_chunkSizeInElementsOrig;
  hsize_t m_maxChunkSizeInBytes;
  int m_minObjectsPerChunk;
  int m_maxObjectsPerChunk;
  hsize_t m_minChunkCacheSize;
  hsize_t m_maxChunkCacheSize;

  // chunk parameters for specific datasets
  hsize_t m_eventIdChunkSizeInBytes;
  int m_eventIdChunkSizeInElements, m_eventIdChunkSizeInElementsOrig;

  hsize_t m_damageChunkSizeInBytes;
  int m_damageChunkSizeInElements, m_damageChunkSizeInElementsOrig;

  hsize_t m_filterMsgChunkSizeInBytes;
  int m_filterMsgChunkSizeInElements, m_filterMsgChunkSizeInElementsOrig;
  
  hsize_t m_epicsPvChunkSizeInBytes;
  int m_epicsPvChunkSizeInElements, m_epicsPvChunkSizeInElementsOrig;

  boost::shared_ptr<Translator::ChunkPolicy> m_defaultChunkPolicy;
  boost::shared_ptr<Translator::ChunkPolicy> m_eventIdChunkPolicy;
  boost::shared_ptr<Translator::ChunkPolicy> m_damageChunkPolicy;
  boost::shared_ptr<Translator::ChunkPolicy> m_filterMsgChunkPolicy;
  boost::shared_ptr<Translator::ChunkPolicy> m_epicsPvChunkPolicy;

  // Copy constructor and assignment are disabled by default
  ChunkManager(const ChunkManager &);
  ChunkManager & operator=(const ChunkManager &);

}; // class ChunkManager

} // namespace Translator 
#endif
