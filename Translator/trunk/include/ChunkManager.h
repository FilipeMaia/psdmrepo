#ifndef TRANSLATOR_CHUNK_MANAGER_H
#define TRANSLATOR_CHUNK_MANAGER_H

#include "PSEnv/Env.h"
#include "Translator/ChunkPolicy.h"

namespace Translator {

class H5Output;

/**
 *  @ingroup Translator
 *
 *  @brief Manages chunk settings and ChunkPolicy classes for Translator.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */

class ChunkManager {
 public:
  ChunkManager();
  ~ChunkManager();
  void readConfigParameters(const Translator::H5Output &, std::list<std::string> &);
  void beginJob(PSEnv::Env &env);
  void beginCalibCycle(PSEnv::Env &env);
  void endCalibCycle(size_t numberEventsInCalibCycle);
  void endJob() {};

  boost::shared_ptr<Translator::ChunkPolicy> eventIdChunkPolicy() { return m_eventIdChunkPolicy; }
  boost::shared_ptr<Translator::ChunkPolicy> damageChunkPolicy() { return m_damageChunkPolicy; }
  boost::shared_ptr<Translator::ChunkPolicy> stringChunkPolicy() { return m_stringChunkPolicy; }
  boost::shared_ptr<Translator::ChunkPolicy> epicsPvChunkPolicy() { return m_epicsPvChunkPolicy; }
  boost::shared_ptr<Translator::ChunkPolicy> defaultChunkPolicy() { return m_defaultChunkPolicy; }
  boost::shared_ptr<Translator::ChunkPolicy> ndarrayChunkPolicy() { return m_ndarrayChunkPolicy; }

 protected:
  void checkForControlDataToSetChunkSize(PSEnv::Env &env);
  void setChunkSizeTargetObjects(int chunkSizeTargetObjects);

 private:
  hsize_t m_chunkSizeTargetInBytes;
  int m_chunkSizeTargetObjects;
  int m_chunkSizeTargetObjectsOrig;
  hsize_t m_maxChunkSizeInBytes;
  int m_minObjectsPerChunk;
  int m_maxObjectsPerChunk;
  int m_chunkCacheSizeTargetInChunks;
  hsize_t m_maxChunkCacheSizeInBytes;

  // chunk parameters for specific datasets
  hsize_t m_eventIdChunkSizeTargetInBytes;
  int m_eventIdChunkSizeTargetObjects;
  int m_eventIdChunkSizeTargetObjectsOrig;

  hsize_t m_damageChunkSizeTargetInBytes;
  int m_damageChunkSizeTargetObjects;
  int m_damageChunkSizeTargetObjectsOrig;

  hsize_t m_stringChunkSizeTargetInBytes;
  int m_stringChunkSizeTargetObjects;
  int m_stringChunkSizeTargetObjectsOrig;
  
  hsize_t m_ndarrayChunkSizeTargetInBytes;
  int m_ndarrayChunkSizeTargetObjects;
  int m_ndarrayChunkSizeTargetObjectsOrig;

  hsize_t m_epicsPvChunkSizeTargetInBytes;
  int m_epicsPvChunkSizeTargetObjects;
  int m_epicsPvChunkSizeTargetObjectsOrig;

  bool m_useControlData;

  boost::shared_ptr<Translator::ChunkPolicy> m_defaultChunkPolicy;
  boost::shared_ptr<Translator::ChunkPolicy> m_eventIdChunkPolicy;
  boost::shared_ptr<Translator::ChunkPolicy> m_damageChunkPolicy;
  boost::shared_ptr<Translator::ChunkPolicy> m_stringChunkPolicy;
  boost::shared_ptr<Translator::ChunkPolicy> m_epicsPvChunkPolicy;
  boost::shared_ptr<Translator::ChunkPolicy> m_ndarrayChunkPolicy;

  // Copy constructor and assignment are disabled by default
  ChunkManager(const ChunkManager &);
  ChunkManager & operator=(const ChunkManager &);

}; // class ChunkManager

} // namespace Translator 
#endif
