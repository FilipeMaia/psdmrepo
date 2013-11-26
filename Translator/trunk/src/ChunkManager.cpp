#include "psddl_psana/control.ddl.h"

#include "Translator/ChunkManager.h"
#include "Translator/H5Output.h"
#include "Translator/ChunkPolicy.h"

using namespace Translator;

namespace {

std::vector<int> nullVectorInt;
std::vector<size_t> nullVectorSize;

const char *logger = "ChunkManager";

// this templatized function is meant to be called with one of the 
// Psana::ControlData::ConfigV* classes, where * is the version.
// if it finds the given controlData and it specifies events, it returns
// the number of events.  Otherwise it returns 0.
// 
template <class T>
unsigned setEventsFromControlCfg(PSEnv::Env &env) {
  boost::shared_ptr<T> cfg = env.configStore().get(PSEvt::Source());
  if (cfg and (cfg->uses_events())) {
    return cfg->events();
  }
  return false;
}

template <>
unsigned setEventsFromControlCfg<Psana::ControlData::ConfigV3>(PSEnv::Env &env) {
  boost::shared_ptr<Psana::ControlData::ConfigV3> cfg = env.configStore().get(PSEvt::Source());
  if (cfg and (cfg->uses_events() or cfg->uses_l3t_events())) {
    return cfg->events();
  }
  return 0;
}

void msgLogStats(const char *dtype, const std::vector<int> &returnedChunkCacheSizes, const std::vector<int> &returnedChunkSizes, const std::vector<size_t> &objSizesDuringChunkCacheCalls) {
  WithMsgLog(logger,debug,str) {
    str << " chunkCache: " << (void *)(&returnedChunkCacheSizes) << " chunks: " << (void *)(&returnedChunkSizes) << " objs: " << (void *)(&objSizesDuringChunkCacheCalls) << std::endl;
    str << dtype << " objsizes:       ";
    for (size_t idx = 0; idx < objSizesDuringChunkCacheCalls.size(); ++idx) str << " " << objSizesDuringChunkCacheCalls[idx];
    str << std::endl;
    str << dtype << " chunkSizes:     ";
    for (size_t idx = 0; idx < returnedChunkSizes.size(); ++idx) str << " " << returnedChunkSizes[idx];
    str << std::endl;
    str << dtype << " chunkCacheSizes:";
    for (size_t idx = 0; idx < returnedChunkCacheSizes.size(); ++idx) str << " " << returnedChunkCacheSizes[idx];
    str << std::endl;
  }
}

} // local namespace


ChunkManager::ChunkManager() 
{
}

ChunkManager::~ChunkManager() 
{
}

void ChunkManager::readConfigParameters(const Translator::H5Output &h5Output) {
  // chunk parameters
  m_chunkSizeInBytes = h5Output.config("chunkSizeInBytes",16*1024*1024);  // default - 16 MB chunks
  m_chunkSizeInElements = h5Output.config("chunkSizeInElements", 0);      // a non-zero value overrides 
  m_chunkSizeInElementsOrig = m_chunkSizeInElements;
  
                                                                 // default chunk size calculation
  m_maxChunkSizeInBytes = h5Output.config("maxChunkSizeInBytes",100*1024*1024); // max chunk size is 100 MB
  m_minObjectsPerChunk = h5Output.config("minObjectsPerChunk",50);              
  m_maxObjectsPerChunk = h5Output.config("maxObjectsPerChunk",2048);

  // the chunk cache needs to be big enough for at least one chunk, otherwise the chunk gets
  // brought back and forth from disk when writting each element.  Ideally the chunk cache holds
  // all the chunks that we are working with for a dataset.
  m_minChunkCacheSize = h5Output.config("minChunkCacheSize",1024*1024);    
  m_maxChunkCacheSize = h5Output.config("maxChunkCacheSize",3*m_maxChunkSizeInBytes);

  m_eventIdChunkSizeInBytes = h5Output.config("eventIdChunkSizeInBytes",16*1024); 
  m_eventIdChunkSizeInElements = h5Output.config("eventIdChunkSizeInElements", m_chunkSizeInElements);
  m_eventIdChunkSizeInElementsOrig = m_eventIdChunkSizeInElements;

  m_damageChunkSizeInBytes = h5Output.config("damageChunkSizeInBytes",m_chunkSizeInBytes);
  m_damageChunkSizeInElements = h5Output.config("damageChunkSizeInElements",m_chunkSizeInElements);
  m_damageChunkSizeInElementsOrig = m_damageChunkSizeInElements;

  m_filterMsgChunkSizeInBytes = h5Output.config("filterMsgChunkSizeInBytes",m_chunkSizeInBytes);
  m_filterMsgChunkSizeInElements = h5Output.config("filterMsgChunkSizeInElements",m_chunkSizeInElements);
  m_filterMsgChunkSizeInElementsOrig = m_filterMsgChunkSizeInElements;

  // there will be a lot of epics datasets, and an epics entry tends to be only 30 bytes or so.
  // If we used 16 Mb chunks, and we had 200 epics pv, we'd have at least 3.2 Gb of chunks that we'd 
  // be writing, and each chunk would hold an hours worth of data.  
  // We set the epics pv chunk size in bytes to 16 kilobytes.
  m_epicsPvChunkSizeInBytes = h5Output.config("epicsPvChunkSizeBytes",16*1024); 
  m_epicsPvChunkSizeInElements = h5Output.config("epicsPvChunkSizeBytes",m_chunkSizeInElements);
  m_epicsPvChunkSizeInElementsOrig = m_epicsPvChunkSizeInElements;

  m_defaultChunkPolicy = boost::make_shared<Translator::ChunkPolicy>(m_chunkSizeInBytes,
                                                                     m_chunkSizeInElements,
                                                                     m_maxChunkSizeInBytes,
                                                                     m_minObjectsPerChunk,
                                                                     m_maxObjectsPerChunk,
                                                                     m_minChunkCacheSize,
                                                                     m_maxChunkCacheSize);

  m_eventIdChunkPolicy = boost::make_shared<Translator::ChunkPolicy>(m_eventIdChunkSizeInBytes,
                                                                     m_eventIdChunkSizeInElements,
                                                                     m_maxChunkSizeInBytes,
                                                                     m_minObjectsPerChunk,
                                                                     m_maxObjectsPerChunk,
                                                                     m_minChunkCacheSize,
                                                                     m_maxChunkCacheSize);

  m_damageChunkPolicy = boost::make_shared<Translator::ChunkPolicy>(m_damageChunkSizeInBytes,
                                                                    m_damageChunkSizeInElements,
                                                                    m_maxChunkSizeInBytes,
                                                                    m_minObjectsPerChunk,
                                                                    m_maxObjectsPerChunk,
                                                                    m_minChunkCacheSize,
                                                                    m_maxChunkCacheSize);

  m_filterMsgChunkPolicy = boost::make_shared<Translator::ChunkPolicy>(m_filterMsgChunkSizeInBytes,
                                                                       m_filterMsgChunkSizeInElements,
                                                                       m_maxChunkSizeInBytes,
                                                                       m_minObjectsPerChunk,
                                                                       m_maxObjectsPerChunk,
                                                                       m_minChunkCacheSize,
                                                                       m_maxChunkCacheSize);
  m_epicsPvChunkPolicy = boost::make_shared<Translator::ChunkPolicy>(m_epicsPvChunkSizeInBytes,
                                                                     m_epicsPvChunkSizeInElements,
                                                                     m_maxChunkSizeInBytes,
                                                                     m_minObjectsPerChunk,
                                                                     m_maxObjectsPerChunk,
                                                                     m_minChunkCacheSize,
                                                                     m_maxChunkCacheSize);
}

void ChunkManager::beginJob(PSEnv::Env &env) {
  checkForControlDataToSetChunkSize(env);
}

void ChunkManager::beginCalibCycle(PSEnv::Env &env) {
  checkForControlDataToSetChunkSize(env);
  clearStats();
}
  
void ChunkManager::endCalibCycle(size_t ) { // numberEventsInCalibCycle) {
  // here is opportunity to try to adjust the 
  // number of events based on what happend last time
  reportStats();
}

void ChunkManager::clearStats() {
  eventIdChunkPolicy()->clearStats();
  damageChunkPolicy()->clearStats();
  filterMsgChunkPolicy()->clearStats();
  epicsPvChunkPolicy()->clearStats();
  defaultChunkPolicy()->clearStats();
}

void ChunkManager::reportStats() {
  const std::vector<int> *returnedChunkCacheSizes = NULL;
  const std::vector<int> *returnedChunkSizes = NULL;
  const std::vector<size_t> *objSizesDuringChunkCacheCalls = NULL;

  eventIdChunkPolicy()->getStats(returnedChunkCacheSizes, returnedChunkSizes, objSizesDuringChunkCacheCalls);
  msgLogStats("eventId",*returnedChunkCacheSizes, *returnedChunkSizes, *objSizesDuringChunkCacheCalls);

  damageChunkPolicy()->getStats(returnedChunkCacheSizes, returnedChunkSizes, objSizesDuringChunkCacheCalls);
  msgLogStats("damage",*returnedChunkCacheSizes, *returnedChunkSizes, *objSizesDuringChunkCacheCalls);

  epicsPvChunkPolicy()->getStats(returnedChunkCacheSizes, returnedChunkSizes, objSizesDuringChunkCacheCalls);
  msgLogStats("epicsPv",*returnedChunkCacheSizes, *returnedChunkSizes, *objSizesDuringChunkCacheCalls);

  defaultChunkPolicy()->getStats(returnedChunkCacheSizes, returnedChunkSizes, objSizesDuringChunkCacheCalls);
  msgLogStats("default",*returnedChunkCacheSizes, *returnedChunkSizes, *objSizesDuringChunkCacheCalls);
}

void ChunkManager::checkForControlDataToSetChunkSize(PSEnv::Env &env) {
  unsigned events = setEventsFromControlCfg<Psana::ControlData::ConfigV3>(env);
  if (not events) {
    events = setEventsFromControlCfg<Psana::ControlData::ConfigV2>(env);
  } 
  if (not events) {
    events = setEventsFromControlCfg<Psana::ControlData::ConfigV1>(env);
  }
  if (not events) {
    MsgLog(logger, trace, "Did not find controlData containing number of events");
    setChunkSizeInElements(0);
  }
  if (events) {
    int chunkSizeInElements = events + events/20 + 10;
    MsgLog(logger, trace, "Found controlData with event number, setting chunkSizeInElements to " 
           << m_chunkSizeInElements);
    setChunkSizeInElements(chunkSizeInElements);
  }
}

void ChunkManager::setChunkSizeInElements(int chunkSizeInElements) {
  if (chunkSizeInElements==0) {
    m_defaultChunkPolicy->chunkSizeInElements(m_chunkSizeInElementsOrig);
    m_eventIdChunkPolicy->chunkSizeInElements(m_eventIdChunkSizeInElementsOrig);
    m_damageChunkPolicy->chunkSizeInElements(m_damageChunkSizeInElementsOrig);
    m_filterMsgChunkPolicy->chunkSizeInElements(m_filterMsgChunkSizeInElementsOrig);
    m_epicsPvChunkPolicy->chunkSizeInElements(m_epicsPvChunkSizeInElementsOrig);
  } else {
    m_defaultChunkPolicy->chunkSizeInElements(m_chunkSizeInElements = chunkSizeInElements);
    m_eventIdChunkPolicy->chunkSizeInElements(m_eventIdChunkSizeInElements = chunkSizeInElements);
    m_damageChunkPolicy->chunkSizeInElements(m_damageChunkSizeInElements = chunkSizeInElements);
    MsgLog(logger,info,"set damage to " << chunkSizeInElements);
    m_filterMsgChunkPolicy->chunkSizeInElements(m_filterMsgChunkSizeInElements = chunkSizeInElements);
    m_epicsPvChunkPolicy->chunkSizeInElements(m_epicsPvChunkSizeInElements = chunkSizeInElements);
  }
}


