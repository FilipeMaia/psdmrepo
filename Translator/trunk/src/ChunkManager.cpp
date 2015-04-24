#include "psddl_psana/control.ddl.h"

#include "Translator/ChunkManager.h"
#include "Translator/H5Output.h"
#include "Translator/ChunkPolicy.h"

using namespace Translator;

namespace {

const char *logger = "ChunkManager";

const hsize_t _16MB = 16*1024*1024;
const hsize_t _100MB = 100*1024*1024;
const hsize_t _16KB = 16*1024;

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

void msgLogStats(const char *dtype, const std::vector<int> &returnedChunkCacheSizesInChunks, const std::vector<int> &returnedChunkSizesInObjects, const std::vector<size_t> &objSizesInBytesDuringChunkCacheCalls) {
  WithMsgLog(logger,debug,str) {
    str << dtype << " objsizes (in bytes):      ";
    for (size_t idx = 0; idx < objSizesInBytesDuringChunkCacheCalls.size(); ++idx) str << " " << objSizesInBytesDuringChunkCacheCalls[idx];
    str << std::endl;
    str << dtype << " chunkSizes (in objects):  ";
    for (size_t idx = 0; idx < returnedChunkSizesInObjects.size(); ++idx) str << " " << returnedChunkSizesInObjects[idx];
    str << std::endl;
    str << dtype << " chunkCacheSizes: (chunks) ";
    for (size_t idx = 0; idx < returnedChunkCacheSizesInChunks.size(); ++idx) str << " " << returnedChunkCacheSizesInChunks[idx];
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

void ChunkManager::readConfigParameters(const Translator::H5Output &h5Output, std::list<std::string> &remainingConfigKeys) {
  // chunk parameters
  m_chunkSizeTargetInBytes = h5Output.configReportIfNotDefault("chunkSizeTargetInBytes",_16MB);  // default - 16 MB chunks
  remainingConfigKeys.remove("chunkSizeTargetInBytes");
  m_chunkSizeTargetObjects = h5Output.configReportIfNotDefault("chunkSizeTargetObjects", 0);      // a non-zero value overrides 
  remainingConfigKeys.remove("chunkSizeTargetObjects");
  m_chunkSizeTargetObjectsOrig = m_chunkSizeTargetObjects;
  
                                                                 // default chunk size calculation
  m_maxChunkSizeInBytes = h5Output.configReportIfNotDefault("maxChunkSizeInBytes",_100MB); // max chunk size is 100 MB
  remainingConfigKeys.remove("maxChunkSizeInBytes");
  m_minObjectsPerChunk = h5Output.configReportIfNotDefault("minObjectsPerChunk",50);              
  remainingConfigKeys.remove("minObjectsPerChunk");
  m_maxObjectsPerChunk = h5Output.configReportIfNotDefault("maxObjectsPerChunk",2048);
  remainingConfigKeys.remove("maxObjectsPerChunk");

  // the chunk cache needs to be big enough for at least one chunk, otherwise the chunk gets
  // brought back and forth from disk when writting each element.  Ideally the chunk cache holds
  // all the chunks that we are working with for a dataset.
  m_chunkCacheSizeTargetInChunks = h5Output.configReportIfNotDefault("chunkCacheSizeTargetInChunks",3);
  remainingConfigKeys.remove("chunkCacheSizeTargetInChunks");
  m_maxChunkCacheSizeInBytes = h5Output.configReportIfNotDefault("maxChunkCacheSizeInBytes",_100MB);
  remainingConfigKeys.remove("maxChunkCacheSizeInBytes");

  m_eventIdChunkSizeTargetInBytes = h5Output.configReportIfNotDefault("eventIdChunkSizeTargetInBytes",_16KB); // 16 KB
  remainingConfigKeys.remove("eventIdChunkSizeTargetInBytes");
  m_eventIdChunkSizeTargetObjects = h5Output.configReportIfNotDefault("eventIdChunkSizeTargetObjects", m_chunkSizeTargetObjects);
  remainingConfigKeys.remove("eventIdChunkSizeTargetObjects");
  m_eventIdChunkSizeTargetObjectsOrig = m_eventIdChunkSizeTargetObjects;

  m_damageChunkSizeTargetInBytes = h5Output.configReportIfNotDefault("damageChunkSizeTargetInBytes",m_chunkSizeTargetInBytes);
  remainingConfigKeys.remove("damageChunkSizeTargetInBytes");
  m_damageChunkSizeTargetObjects = h5Output.configReportIfNotDefault("damageChunkSizeTargetObjects",m_chunkSizeTargetObjects);
  remainingConfigKeys.remove("damageChunkSizeTargetObjects");
  m_damageChunkSizeTargetObjectsOrig = m_damageChunkSizeTargetObjects;

  m_stringChunkSizeTargetInBytes = h5Output.configReportIfNotDefault("stringChunkSizeTargetInBytes",m_chunkSizeTargetInBytes);
  remainingConfigKeys.remove("stringChunkSizeTargetInBytes");
  m_stringChunkSizeTargetObjects = h5Output.configReportIfNotDefault("stringChunkSizeTargetObjects",m_chunkSizeTargetObjects);
  remainingConfigKeys.remove("stringChunkSizeTargetObjects");
  m_stringChunkSizeTargetObjectsOrig = m_stringChunkSizeTargetObjects;

  m_ndarrayChunkSizeTargetInBytes = h5Output.configReportIfNotDefault("ndarrayChunkSizeTargetInBytes",m_chunkSizeTargetInBytes);
  remainingConfigKeys.remove("ndarrayChunkSizeTargetInBytes");
  m_ndarrayChunkSizeTargetObjects = h5Output.configReportIfNotDefault("ndarrayChunkSizeTargetObjects",m_chunkSizeTargetObjects);
  remainingConfigKeys.remove("ndarrayChunkSizeTargetObjects");
  m_ndarrayChunkSizeTargetObjectsOrig = m_ndarrayChunkSizeTargetObjects;

  // there will be a lot of epics datasets, and an epics entry tends to be only 30 bytes or so.
  // If we used 16 Mb chunks, and we had 200 epics pv, we'd have at least 3.2 Gb of chunks that we'd 
  // be writing, and each chunk would hold an hours worth of data.  
  // We set the epics pv chunk size in bytes to 16 kilobytes.  This is not the right thing to do for 
  // epics pv's that contain projections or image data. Below is just for pv's with a few values.
  m_epicsPvChunkSizeTargetInBytes = h5Output.configReportIfNotDefault("epicsPvChunkSizeTargetInBytes",_16KB); 
  remainingConfigKeys.remove("epicsPvChunkSizeTargetInBytes");
  m_epicsPvChunkSizeTargetObjects = h5Output.configReportIfNotDefault("epicsPvChunkSizeTargetObjects",m_chunkSizeTargetObjects);
  remainingConfigKeys.remove("epicsPvChunkSizeTargetObjects");
  m_epicsPvChunkSizeTargetObjectsOrig = m_epicsPvChunkSizeTargetObjects;
  m_useControlData = h5Output.configReportIfNotDefault("useControlData",true);
  remainingConfigKeys.remove("useControlData");

  m_defaultChunkPolicy = boost::make_shared<Translator::ChunkPolicy>(m_chunkSizeTargetInBytes,
                                                                     m_chunkSizeTargetObjects,
                                                                     m_maxChunkSizeInBytes,
                                                                     m_minObjectsPerChunk,
                                                                     m_maxObjectsPerChunk,
                                                                     m_chunkCacheSizeTargetInChunks,
                                                                     m_maxChunkCacheSizeInBytes);

  m_eventIdChunkPolicy = boost::make_shared<Translator::ChunkPolicy>(m_eventIdChunkSizeTargetInBytes,
                                                                     m_eventIdChunkSizeTargetObjects,
                                                                     m_maxChunkSizeInBytes,
                                                                     m_minObjectsPerChunk,
                                                                     m_maxObjectsPerChunk,
                                                                     m_chunkCacheSizeTargetInChunks,
                                                                     m_maxChunkCacheSizeInBytes);

  m_damageChunkPolicy = boost::make_shared<Translator::ChunkPolicy>(m_damageChunkSizeTargetInBytes,
                                                                    m_damageChunkSizeTargetObjects,
                                                                    m_maxChunkSizeInBytes,
                                                                    m_minObjectsPerChunk,
                                                                    m_maxObjectsPerChunk,
                                                                    m_chunkCacheSizeTargetInChunks,
                                                                    m_maxChunkCacheSizeInBytes);

  m_stringChunkPolicy = boost::make_shared<Translator::ChunkPolicy>(m_stringChunkSizeTargetInBytes,
                                                                       m_stringChunkSizeTargetObjects,
                                                                       m_maxChunkSizeInBytes,
                                                                       m_minObjectsPerChunk,
                                                                       m_maxObjectsPerChunk,
                                                                       m_chunkCacheSizeTargetInChunks,
                                                                       m_maxChunkCacheSizeInBytes);

  m_ndarrayChunkPolicy = boost::make_shared<Translator::ChunkPolicy>(m_ndarrayChunkSizeTargetInBytes,
                                                                       m_ndarrayChunkSizeTargetObjects,
                                                                       m_maxChunkSizeInBytes,
                                                                       m_minObjectsPerChunk,
                                                                       m_maxObjectsPerChunk,
                                                                       m_chunkCacheSizeTargetInChunks,
                                                                       m_maxChunkCacheSizeInBytes);

  m_epicsPvChunkPolicy = boost::make_shared<Translator::ChunkPolicy>(m_epicsPvChunkSizeTargetInBytes,
                                                                     m_epicsPvChunkSizeTargetObjects,
                                                                     m_maxChunkSizeInBytes,
                                                                     m_minObjectsPerChunk,
                                                                     m_maxObjectsPerChunk,
                                                                     m_chunkCacheSizeTargetInChunks,
                                                                     m_maxChunkCacheSizeInBytes);
}

void ChunkManager::beginJob(PSEnv::Env &env) {
  if (m_useControlData) {
    checkForControlDataToSetChunkSize(env);
  }
}

void ChunkManager::beginCalibCycle(PSEnv::Env &env) {
  if (m_useControlData) {
  checkForControlDataToSetChunkSize(env);
  }
}
  
void ChunkManager::endCalibCycle(size_t ) { // numberEventsInCalibCycle) {
  // here is opportunity to try to adjust the 
  // number of events based on what happend last time
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
    setChunkSizeTargetObjects(0);
  }
  if (events) {
    int chunkSizeTargetObjects = events + events/20 + 10;
    MsgLog(logger, trace, "Found controlData with event number, setting chunkSizeTargetObjects to " 
           << chunkSizeTargetObjects);
    setChunkSizeTargetObjects(chunkSizeTargetObjects);
  }
}

void ChunkManager::setChunkSizeTargetObjects(int chunkSizeTargetObjects) {
  if (chunkSizeTargetObjects==0) {
    m_defaultChunkPolicy->chunkSizeTargetObjects(m_chunkSizeTargetObjectsOrig);
    m_eventIdChunkPolicy->chunkSizeTargetObjects(m_eventIdChunkSizeTargetObjectsOrig);
    m_damageChunkPolicy->chunkSizeTargetObjects(m_damageChunkSizeTargetObjectsOrig);
    m_stringChunkPolicy->chunkSizeTargetObjects(m_stringChunkSizeTargetObjectsOrig);
    m_ndarrayChunkPolicy->chunkSizeTargetObjects(m_ndarrayChunkSizeTargetObjectsOrig);
    m_epicsPvChunkPolicy->chunkSizeTargetObjects(m_epicsPvChunkSizeTargetObjectsOrig);
  } else {
    m_defaultChunkPolicy->chunkSizeTargetObjects(m_chunkSizeTargetObjects = chunkSizeTargetObjects);
    m_eventIdChunkPolicy->chunkSizeTargetObjects(m_eventIdChunkSizeTargetObjects = chunkSizeTargetObjects);
    m_damageChunkPolicy->chunkSizeTargetObjects(m_damageChunkSizeTargetObjects = chunkSizeTargetObjects);
    m_stringChunkPolicy->chunkSizeTargetObjects(m_stringChunkSizeTargetObjects = chunkSizeTargetObjects);
    m_ndarrayChunkPolicy->chunkSizeTargetObjects(m_ndarrayChunkSizeTargetObjects = chunkSizeTargetObjects);
    m_epicsPvChunkPolicy->chunkSizeTargetObjects(m_epicsPvChunkSizeTargetObjects = chunkSizeTargetObjects);
  }
}


