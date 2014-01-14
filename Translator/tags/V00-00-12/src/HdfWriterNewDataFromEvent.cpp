#include "Translator/HdfWriterNewDataFromEvent.h"

namespace {
  const char *logger = "HdfWriterNewData";
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

void HdfWriterNewDataFromEvent::make_datasets(DataTypeLoc dataTypeLoc,
                                              hdf5pp::Group & srcGroup, 
                                              const PSEvt::EventKey & eventKey, 
                                              PSEvt::Event & evt, 
                                              PSEnv::Env & env,
                                              bool shuffle,
                                              int deflate,
                                              boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy)
{
  if (not (*(eventKey.typeinfo()) == *m_typeInfoPtr)) {
    MsgLog(logger, fatal, " make_datsets for new data, but eventkey type does not agree with stored type"
           << " stored: " << PSEvt::TypeInfoUtils::typeInfoRealName(m_typeInfoPtr)
           << " eventKey type: " << PSEvt::TypeInfoUtils::typeInfoRealName(eventKey.typeinfo()));
  }
  boost::shared_ptr<void> data;
  if (dataTypeLoc == inEvent) {
    data = evt.proxyDict()->get(eventKey.typeinfo(), PSEvt::Source(eventKey.src()), eventKey.key(), NULL); 
  } else if (dataTypeLoc == inConfigStore) {
    data = env.configStore().proxyDict()->get(eventKey.typeinfo(), PSEvt::Source(eventKey.src()), eventKey.key(), NULL); 
  }
  if (not data) {
    MsgLog(logger,error, "could not retrieve data from event store, eventkey=" 
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
    MsgLog(logger, fatal, "HdfWriterNewDataFromEvent, H5Tget_size call failed for h5type="
           << h5type << " returned for C++ type: " << PSEvt::TypeInfoUtils::typeInfoRealName(m_typeInfoPtr));
  }
  
  Translator::DataSetCreationProperties dataSetCreationProperties(chunkPolicy,shuffle, deflate);
  m_writer.createUnlimitedSizeDataset(srcGroup.id(), 
                                      m_datasetName,
                                      h5type,
                                      dataSetCreationProperties);
}

void HdfWriterNewDataFromEvent::append(DataTypeLoc dataTypeLoc,
                                       hdf5pp::Group & srcGroup, 
                                       const PSEvt::EventKey & eventKey, 
                                       PSEvt::Event & evt, 
                                       PSEnv::Env & env)
{
  if (not (*(eventKey.typeinfo()) == *m_typeInfoPtr)) {
    MsgLog(logger, fatal, " append for new data, but eventkey type does not agree with stored type"
           << " stored: " << PSEvt::TypeInfoUtils::typeInfoRealName(m_typeInfoPtr)
           << " eventKey type: " << PSEvt::TypeInfoUtils::typeInfoRealName(eventKey.typeinfo()));
  }
  boost::shared_ptr<void> data;
  if (dataTypeLoc == inEvent) {
    data = evt.proxyDict()->get(eventKey.typeinfo(), PSEvt::Source(eventKey.src()), eventKey.key(), NULL); 
  } else if (dataTypeLoc == inConfigStore) {
    data = env.configStore().proxyDict()->get(eventKey.typeinfo(), PSEvt::Source(eventKey.src()), eventKey.key(), NULL); 
  }
  if (not data) {
    MsgLog(logger,error, "could not retrieve data from event store, eventkey=" 
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
  throw NotImplementedException(ERR_LOC, "HdfWriterNewDataFromEvent::store()");
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
