#include "Translator/HdfWriterNewDataFromEvent.h"

namespace {
  const char *logger = "HdfWriterNewData";

  boost::shared_ptr<void> getData(Translator::DataTypeLoc dataTypeLoc, const PSEvt::EventKey & eventKey,
                                  PSEvt::Event & evt, PSEnv::Env & env)
  {
    if (dataTypeLoc == Translator::inEvent) {
      return evt.proxyDict()->get(eventKey.typeinfo(), 
                                  PSEvt::Source(eventKey.src()), eventKey.key(), NULL); 
    } else if (dataTypeLoc == Translator::inConfigStore) {
      return env.configStore().proxyDict()->get(eventKey.typeinfo(), 
                                                PSEvt::Source(eventKey.src()), eventKey.key(), NULL);
    } else if (dataTypeLoc == Translator::inCalibStore) {
      return env.calibStore().proxyDict()->get(eventKey.typeinfo(), 
                                               PSEvt::Source(eventKey.src()), eventKey.key(), NULL);
    }
    MsgLog(logger,error,"getData: dataTypeLoc not recognized: " << dataTypeLoc << " for eventKey: " << eventKey);
    boost::shared_ptr<void> nullPtr;
    return nullPtr;
  }
  
}

namespace Translator {

HdfWriterNewDataFromEvent::HdfWriterNewDataFromEvent(const HdfWriterNew & newWriter,
                                                     const std::string &key)
  :  m_typeInfoPtr(newWriter.typeInfoPtr()),
     m_datasetName(newWriter.datasetName()),
     m_createType(newWriter.createType()),
     m_fillWriteBuffer(newWriter.fillWriteBuffer()),
     m_closeType( newWriter.closeType()),
     m_writer("newWriter_" + key),
     m_key(key)
{
}

bool HdfWriterNewDataFromEvent::checkTypeMatch(const PSEvt::EventKey & eventKey, std::string msg)
{
  if (*(eventKey.typeinfo()) == *m_typeInfoPtr) return true;

  MsgLog(logger, error,  msg + " eventkey type does not agree with stored type"
         << " stored: " << PSEvt::TypeInfoUtils::typeInfoRealName(m_typeInfoPtr)
         << " eventKey type: " << PSEvt::TypeInfoUtils::typeInfoRealName(eventKey.typeinfo()));
  return false;
}

void HdfWriterNewDataFromEvent::make_datasets(DataTypeLoc dataTypeLoc,
                                              hdf5pp::Group & srcGroup, 
                                              const PSEvt::EventKey & eventKey, 
                                              PSEvt::Event & evt, 
                                              PSEnv::Env & env,
                                              bool shuffle,
                                              int deflate,
                                              boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy)
{
  if (not checkTypeMatch(eventKey)) return;

  boost::shared_ptr<void> data =  getData(dataTypeLoc, eventKey, evt, env);
  if (not data) {
    MsgLog(logger,error, "make_datasets: could not retrieve eventkey=" 
           << eventKey << "loc=" << dataTypeLoc);
    return;
  }
  if (m_createType == NULL) {
    MsgLog(logger,error, "newWriter createType is NULL, key=" << m_key);
    return;
  }
  hid_t h5type = m_createType(data.get());
  size_t size = H5Tget_size(h5type);
  if ( size == 0 ) {
    MsgLog(logger, error, "HdfWriterNewDataFromEvent, H5Tget_size call failed for h5type="
           << h5type << " returned for C++ type: " << PSEvt::TypeInfoUtils::typeInfoRealName(m_typeInfoPtr));
    return;
  }
  
  Translator::DataSetCreationProperties dataSetCreationProperties(chunkPolicy,shuffle, deflate);
  m_writer.createUnlimitedSizeDataset(srcGroup.id(), 
                                      m_datasetName,
                                      h5type, h5type,
                                      dataSetCreationProperties);
}

void HdfWriterNewDataFromEvent::append(DataTypeLoc dataTypeLoc,
                                       hdf5pp::Group & srcGroup, 
                                       const PSEvt::EventKey & eventKey, 
                                       PSEvt::Event & evt, 
                                       PSEnv::Env & env)
{
  if (not checkTypeMatch(eventKey)) return;

  boost::shared_ptr<void> data =  getData(dataTypeLoc, eventKey, evt, env);
  if (not data) {
    MsgLog(logger,error, "append: could not retrieve eventkey=" 
           << eventKey << "loc=" << dataTypeLoc);
    return;
  }

  if (m_fillWriteBuffer == NULL) {
    MsgLog(logger,error, "newWriter fillWriteBuffer is NULL, key=" << m_key);
    return;
  }
  const void * writeBuffer = m_fillWriteBuffer(data.get());
  m_writer.append(srcGroup.id(), m_datasetName, writeBuffer);
}

void HdfWriterNewDataFromEvent::closeDatasets(hdf5pp::Group &group)
{  
  std::map<std::string,hid_t> dset2hid = m_writer.getDatasetNameToH5TypeMap(group.id());  
  if (dset2hid.find(m_datasetName) == dset2hid.end()) {
    MsgLog(logger, error, "unexpected: dsetname " << m_datasetName 
           << " not in generic writer group map, group id=" 
           << group.id());
    return;
  }
  hid_t h5type = dset2hid[m_datasetName];
  m_writer.closeDatasets(group.id());
  if (m_closeType != NULL) {
    m_closeType(h5type);
  }
}


void HdfWriterNewDataFromEvent::store(DataTypeLoc dataTypeLoc,
                                      hdf5pp::Group & srcGroup, 
                                      const PSEvt::EventKey & eventKey, 
                                      PSEvt::Event & evt, 
                                      PSEnv::Env & env)
{
  if (not checkTypeMatch(eventKey)) return;

  boost::shared_ptr<void> data =  getData(dataTypeLoc, eventKey, evt, env);
  if (not data) {
    MsgLog(logger,error, "store: could not retrieve eventkey=" 
           << eventKey << "loc=" << dataTypeLoc);
    return;
  }

  if (m_createType == NULL) {
    MsgLog(logger,error, "newWriter createType is NULL, key=" << m_key);
    return;
  }

  hid_t h5type = m_createType(data.get());
  size_t size = H5Tget_size(h5type);
  if ( size == 0 ) {
    MsgLog(logger, error, "HdfWriterNewDataFromEvent, H5Tget_size call failed for h5type="
           << h5type << " returned for C++ type: " << PSEvt::TypeInfoUtils::typeInfoRealName(m_typeInfoPtr));
    return;
  }
  
  if (m_fillWriteBuffer == NULL) {
    MsgLog(logger,error, "newWriter fillWriteBuffer is NULL, key=" << m_key);
    return;
  }
  const void * writeBuffer = m_fillWriteBuffer(data.get());

  m_writer.createAndStoreDataset(srcGroup.id(), 
                                 m_datasetName,
                                 h5type, h5type,
                                 writeBuffer);
}

void HdfWriterNewDataFromEvent::store_at(DataTypeLoc dataTypeLoc,
                                         long index, hdf5pp::Group & srcGroup, 
                                         const PSEvt::EventKey & eventKey, 
                                         PSEvt::Event & evt, 
                                         PSEnv::Env & env)
{
  throw NotImplementedException(ERR_LOC, "HdfWriterNewDataFromEvent::store_at()");
}


void HdfWriterNewDataFromEvent::addBlank(hdf5pp::Group & group)
{
  throw NotImplementedException(ERR_LOC, "HdfWriterNewDataFromEvent::addBlank()");
}


HdfWriterNewDataFromEvent::~HdfWriterNewDataFromEvent() 
{
}

} // namespace
