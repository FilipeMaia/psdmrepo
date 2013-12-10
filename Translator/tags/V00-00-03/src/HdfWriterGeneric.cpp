#include <sstream>
#include <string>

#include "Translator/HdfWriterGeneric.h"
#include "MsgLogger/MsgLogger.h"
#include "Translator/hdf5util.h"
#include "Translator/firstPrimeGreaterOrEqualTo.h"

using namespace std;
using namespace Translator;

namespace {

string logger(string addTo="") {
  const string base = "Translator.HdfWriterGeneric";
  return (addTo.size()>0) ? base + string(".") + addTo : base;
}

hid_t createPropertyList(const int rank, const DataSetCreationProperties & dsetCreateProp, hid_t typeId, string & debugMsg) {
  size_t typeSize = H5Tget_size(typeId);
  hid_t propId = H5Pcreate(H5P_DATASET_CREATE);
  hsize_t chunkSize = dsetCreateProp.chunkPolicy()->chunkSize(typeSize);
  herr_t setChunkCall = H5Pset_chunk(propId, rank, &chunkSize);
  herr_t shuffleCall=0, deflateCall=0;
  if (dsetCreateProp.shuffle()) shuffleCall = H5Pset_shuffle(propId);
  if (dsetCreateProp.deflate() >= 0) deflateCall = H5Pset_deflate(propId,dsetCreateProp.deflate());
  ostringstream msg;

  // check for errors
  if (typeSize==0) {
    msg << "createPropertyList bad input, size==0 for typeId= " << typeId;
    throw HdfWriterGeneric::PropertyListException(ERR_LOC, msg.str());
  }
  if (propId == -1) throw HdfWriterGeneric::PropertyListException(ERR_LOC,"H5Pcreate call failed");
  msg <<  "chunkSize= " << chunkSize 
      << " typeId=" << typeId << " typeId.size=" << typeSize;
  if (chunkSize<=0) {
    throw HdfWriterGeneric::PropertyListException(ERR_LOC, msg.str());
  }
  if (setChunkCall < 0 or shuffleCall < 0 or deflateCall < 0) {
    throw HdfWriterGeneric::PropertyListException(ERR_LOC,
              "error with one or more of H5Pcreate, H5Pset_chunk, H5Pset_shuffle, H5Pset_deflate");
  }

  debugMsg = msg.str();
  return propId;
}

hid_t createDatasetAccessList(const DataSetCreationProperties & dsetCreateProp, hid_t typeId, string & debugMsg) {
  size_t typeSize = H5Tget_size(typeId);
  hid_t datasetAccessId = H5Pcreate(H5P_DATASET_ACCESS);  
  int chunkCacheSize = dsetCreateProp.chunkPolicy()->chunkCacheSize(typeSize);
  int chunkSize = dsetCreateProp.chunkPolicy()->chunkSize(typeSize);
  size_t rdcc_nelmts = firstPrimeGreaterOrEqualTo(chunkCacheSize * 30);
  size_t rdcc_nbytes = chunkCacheSize * typeSize * chunkSize;
  double rdcc_w0 = 0.75;
  herr_t cacheErr = H5Pset_chunk_cache(datasetAccessId, rdcc_nelmts, rdcc_nbytes, rdcc_w0);
  
  ostringstream msg;
  msg << " chunkCacheSize= " << chunkCacheSize 
      << " rdcc_nelmts= " << rdcc_nelmts << " rdcc_nbytes= " << rdcc_nbytes
      << " rdcc_w0= " << rdcc_w0;
  if (typeSize==0 or cacheErr<0) {
    throw HdfWriterGeneric::PropertyListException(ERR_LOC,msg.str());
  }
  debugMsg = msg.str();
  return datasetAccessId;
}

};

HdfWriterGeneric::HdfWriterGeneric(const string &debugName) : m_debugName(debugName) {
  hsize_t current_dims = 1; 
  m_singleTransferDataSpaceIdMemory = H5Screate_simple(m_rankOne, &current_dims, &current_dims );
  if (m_singleTransferDataSpaceIdMemory < 0) {
    throw DataSpaceException(ERR_LOC,"HdfWriterGeneric constructor, Failed to create memory dataspace");
  }
  current_dims = 0;
  hsize_t maximum_dims = H5S_UNLIMITED;
  m_unlimitedDataSpaceIdForFile = H5Screate_simple(m_rankOne, &current_dims, &maximum_dims );
  if (m_unlimitedDataSpaceIdForFile < 0) {
    throw DataSpaceException(ERR_LOC,"HdfWriterGeneric constructor, Failed to create file dataspace");
  }
}

HdfWriterGeneric::~HdfWriterGeneric() {
  herr_t closeMem  = H5Sclose(m_singleTransferDataSpaceIdMemory);
  herr_t closeFile = H5Sclose(m_unlimitedDataSpaceIdForFile);
  if (closeMem < 0 or closeFile < 0) {
    throw DataSpaceException(ERR_LOC,"HdfWriterGeneric desctructor, Failed to close memory or file dataspace");
  }
}

size_t HdfWriterGeneric::createUnlimitedSizeDataset(hid_t groupId,
                                                    const string & dsetName,
                                                    hid_t h5type,
                                                    const DataSetCreationProperties & dsetCreateProp)
{
  size_t dsetIndex = createNewDatasetSlotForGroup(groupId, dsetName);
  string debugMsgPropId, debugMsgAccessId;
  hid_t propId = createPropertyList(m_rankOne, dsetCreateProp, h5type, debugMsgPropId);
  hid_t datasetAccessId = createDatasetAccessList(dsetCreateProp, h5type, debugMsgAccessId);
  MsgLog(logger(),debug,"HdfWriterGeneric::createUnlimitedSizeDataset (" << m_debugName << "): " 
         << debugMsgPropId << " " << debugMsgAccessId << " groupId= " << groupId 
         << " " << hdf5util::objectName(groupId) << " dsetName= " << dsetName);
  hid_t dataset = H5Dcreate(groupId, dsetName.c_str(), h5type, 
                            m_unlimitedDataSpaceIdForFile,
                            H5P_DEFAULT, // link creation property list
                            propId,
                            datasetAccessId);
  if (dataset<0) {
    std::ostringstream msg;
    msg << "HdfWriterGeneric (" << m_debugName << "): failed to make dataset: groupId = " << groupId 
        << " groupName = " << hdf5util::objectName(groupId)<< " dsetname= " << dsetName.c_str();
    throw DataSetException(ERR_LOC,msg.str());
  }
  herr_t statusProp = H5Pclose(propId);
  herr_t statusAccess = H5Pclose(datasetAccessId);
  if (statusProp<0 or statusAccess<0) {
    std::ostringstream msg;
    msg << "HdfWriterGeneric: H5Pclose for dataset creation properties or "
        << "access failed after closing dataset for groupId = " << groupId 
        << " groupName = " << hdf5util::objectName(groupId)<< " dsetname= " << dsetName.c_str();
    throw DataSetException(ERR_LOC, msg.str());
  }
  m_datasetMap[groupId].at(dsetIndex) = DataSetMeta(dsetName,dataset,DataSetMeta::Unlimited,h5type);
  MsgLog(logger(),debug,"createUnlimitedSizeDataset (" << m_debugName 
         << ") added groupId=" << groupId << " " << hdf5util::objectName(groupId) 
         << " to datasetmap, for h5type= " << h5type << " dsetName = " 
         << dsetName << " returning datasetIndex= " << dsetIndex);
  return dsetIndex;
}

size_t HdfWriterGeneric::createFixedSizeDataset(hid_t groupId,
                                                const string & dsetName,
                                                hid_t h5type,
                                                hsize_t fixedSize)
{
  size_t dsetIndex = createNewDatasetSlotForGroup(groupId, dsetName);
  hid_t fixedDataSpaceIdForCreation = H5Screate_simple(m_rankOne, &fixedSize, &fixedSize);
  if (fixedDataSpaceIdForCreation < 0) {
   ostringstream msg;
    msg << "Failed to create fixed size dataspace id for size " << fixedSize
        << " groupId " << groupId << " " << hdf5util::objectName(groupId) 
        << " dsetname: " << dsetName;
    throw DataSetException(ERR_LOC, msg.str());
  }
  hid_t dataset = H5Dcreate(groupId, dsetName.c_str(), h5type, 
                            fixedDataSpaceIdForCreation,
                            H5P_DEFAULT, // link creation property list
                            H5P_DEFAULT,
                            H5P_DEFAULT);  // access
  if (dataset<0) throw DataSetException(ERR_LOC,"error creating dataset");
  m_datasetMap[groupId].at(dsetIndex) = DataSetMeta(dsetName,dataset,DataSetMeta::Fixed,h5type);
  herr_t status = H5Sclose(fixedDataSpaceIdForCreation);
  if (status<0) {
    ostringstream msg;
    msg <<  "error closing dataspace for file creation for fixed size " 
        << fixedSize << " dsetname: " << dsetName;
    throw DataSetException(ERR_LOC,msg.str());
  }
  return dsetIndex;
}

void HdfWriterGeneric::append(hid_t groupId, const string & dsetName, const void * data) 
{  
  std::map<hid_t, vector<DataSetMeta> >::iterator pos = m_datasetMap.find(groupId);
  if (pos == m_datasetMap.end()) {
    ostringstream msg;
    msg << "group id = " << groupId 
        << " " << hdf5util::objectName(groupId) << "  not in map";
    throw GroupMapException(ERR_LOC, msg.str());
  }
  vector<DataSetMeta> & dsetList = pos->second;
  for (size_t idx = 0; idx < dsetList.size(); ++idx) {
    if (dsetList[idx].name() == dsetName) {
      append(groupId, idx, data);
      return;
    }
  }
  ostringstream msg;
  msg << "dset name " << dsetName << " not in list for group = " << groupId 
      << " " << hdf5util::objectName(groupId);
  throw GroupMapException(ERR_LOC, msg.str());
}

void HdfWriterGeneric::store_at(hid_t groupId, long storeIndex, const string & dsetName, const void * data) 
{  
  std::map<hid_t, vector<DataSetMeta> >::iterator pos = m_datasetMap.find(groupId);
  if (pos == m_datasetMap.end()) {
    ostringstream msg;
    msg << "group id = " << groupId << "  not in map";
    throw GroupMapException(ERR_LOC, msg.str());
  }
  vector<DataSetMeta> & dsetList = pos->second;
  for (size_t dsetIdx = 0; dsetIdx < dsetList.size(); ++dsetIdx) {
    if (dsetList[dsetIdx].name() == dsetName) {
      store_at(groupId, storeIndex, dsetIdx,data);
      return;
    }
  }
  ostringstream msg;
  msg << "dset name " << dsetName << " not in list for group = " << groupId;
  throw GroupMapException(ERR_LOC, msg.str());
}

void HdfWriterGeneric::store_at(hid_t groupId, long storeIndex, size_t dsetIndex, const void * data) 
{  
  std::map<hid_t, vector<DataSetMeta> >::iterator pos = m_datasetMap.find(groupId);
  if (pos == m_datasetMap.end()) {
    ostringstream msg;
    msg << "group id " << groupId << " not in map";
    throw GroupMapException(ERR_LOC, msg.str());
  }
  vector<DataSetMeta> & dsetList = pos->second;
  if (dsetIndex >= dsetList.size()) {
    ostringstream msg;
    msg << "dsetIdx=" << dsetIndex  << " outside bounds for groupId= ";
    msg << groupId << " which is [0," <<  dsetList.size() << ")";
    throw GroupMapException(ERR_LOC, msg.str());
  }
  DataSetMeta &dsetMeta = dsetList.at(dsetIndex);
  hid_t datasetId = dsetMeta.dsetId();
  hsize_t currentSize = dsetMeta.currentSize();
  if ((storeIndex>0) and (hsize_t(storeIndex) >= currentSize)) {
    std::ostringstream msg;
    msg << "store_at, storeIndex exceeds size: "
        << "storeIdx = " << storeIndex << " currentSize=" << currentSize
        << " dsetIndex=" << dsetIndex << " groupId=" << groupId;
    throw WriteException(ERR_LOC,msg.str());
  }
  hsize_t start = storeIndex;
  if (storeIndex == -1) {
    hsize_t newSize = currentSize + 1;
    DataSetPos::MaxSize maxSize = dsetMeta.maxSize();
    if (maxSize == DataSetPos::Unlimited) {
      herr_t err = H5Dset_extent(datasetId, &newSize);
      if (err<0) {
        ostringstream msg;
        msg << "failed to increase extent to " << newSize 
            << " for dset=" << datasetId << ", name= " << dsetMeta.name() 
            << " group=" << groupId;
        throw WriteException(ERR_LOC, msg.str());
      }
    }
    dsetMeta.increaseSizeByOne();
    start = currentSize;
  }
  hid_t dspaceFileWriteId = H5Dget_space(datasetId);
  hsize_t count = 1;
  herr_t err = H5Sselect_hyperslab(dspaceFileWriteId,
                                   H5S_SELECT_SET,
                                   &start,
                                   NULL,
                                   &count,
                                   NULL);
  if (err<0) {
    ostringstream msg;
    msg <<  "failed to select hyperslab start=" << start << 
      " for dset=" << datasetId << " name= " << dsetMeta.name() 
        << ", group=" << groupId; 
    throw WriteException(ERR_LOC, msg.str());
  }
  hid_t h5typeId = dsetMeta.typeId();
  err = std::min(0,H5Dwrite(datasetId, 
                            h5typeId, 
                            m_singleTransferDataSpaceIdMemory, 
                            dspaceFileWriteId,
                            H5P_DEFAULT, 
                            data));
  err = std::min(err,H5Sclose(dspaceFileWriteId));
  if(err < 0) throw WriteException(ERR_LOC, "H5Dwrite or H5Sclose failed");
}


void HdfWriterGeneric::closeDatasets(hid_t groupId)
{
  std::map<hid_t, vector<DataSetMeta> >::iterator pos = m_datasetMap.find(groupId);
  if (pos == m_datasetMap.end()) {
    ostringstream msg;
    msg << "close: group id " << groupId 
        << " " << hdf5util::objectName(groupId) << " not in map";
    throw GroupMapException(ERR_LOC,msg.str());  
  }
  vector<DataSetMeta> & dsetList = pos->second;
  for (size_t idx = 0; idx < dsetList.size(); ++idx) {
    DataSetMeta & dsetMeta = dsetList[idx];
    hid_t datasetId = dsetMeta.dsetId();
    herr_t err = H5Dclose(datasetId);
    if (err < 0) {
      ostringstream msg;
      msg << "problem closing dataset: " << datasetId 
          << " name: " << dsetMeta.name() << " for group " << groupId
          << " " << hdf5util::objectName(groupId);
      throw DataSetException(ERR_LOC, msg.str());
    }
  }
  dsetList.clear();
  MsgLog(logger(), debug, "HdfWriterGeneric::closeDatasets (" << m_debugName << ") closed group " 
         << groupId << " " << hdf5util::objectName(groupId));
  m_datasetMap.erase(pos);
}

void HdfWriterGeneric::closeDatasetsForAllGroups() {
  std::map<hid_t, vector<DataSetMeta> >::iterator pos;
  for (pos = m_datasetMap.begin(); pos != m_datasetMap.end(); ++pos) {
    vector<DataSetMeta> & dsetList = pos->second;
    for (size_t idx = 0; idx < dsetList.size(); ++idx) {
      DataSetMeta & dsetMeta = dsetList[idx];
      hid_t datasetId = dsetMeta.dsetId();
      herr_t err = H5Dclose(datasetId);
      if (err < 0) {
        ostringstream msg;
        msg << "closeDatasetsForAllGroups: problem closing dataset: " << datasetId 
            << " name: " << dsetMeta.name() << " for group " << pos->first
            << " " << hdf5util::objectName(pos->first);
        throw DataSetException(ERR_LOC, msg.str());
      }
    }
    dsetList.clear();
  }
  m_datasetMap.clear();
}

hid_t HdfWriterGeneric::getDatasetId(hid_t groupId, size_t dsetIndex) {
  map<hid_t, vector<DataSetMeta> >::iterator pos = m_datasetMap.find(groupId);
  ostringstream msg;
  if (pos == m_datasetMap.end()) {
    msg << "getDatasetID: groupId= " << groupId 
        << " " << hdf5util::objectName(groupId)
        << " not in the map";
    throw GroupMapException(ERR_LOC, msg.str());
  }
  const vector<DataSetMeta> dataSetsMeta = pos->second;
  if (dsetIndex > dataSetsMeta.size()) {
    msg << "getDataSetId: groupId= " << groupId 
        << " " << hdf5util::objectName(groupId)
        << " dsetIndex= " << dsetIndex
        << " is greater than number of datasets for this group which is "
        << dataSetsMeta.size();
    throw GroupMapException(ERR_LOC, msg.str());
  }
  const DataSetMeta & dataSetMeta = dataSetsMeta.at(dsetIndex);
  return dataSetMeta.dsetId();
}

hid_t HdfWriterGeneric::getDatasetId(hid_t groupId, const std::string &dsetName) {
  map<hid_t, vector<DataSetMeta> >::iterator pos = m_datasetMap.find(groupId);
  ostringstream msg;
  if (pos == m_datasetMap.end()) {
    msg << "getDatasetID: groupId= " << groupId 
        << " " << hdf5util::objectName(groupId)
        << " not in the map";
    throw GroupMapException(ERR_LOC, msg.str());
  }
  const vector<DataSetMeta> dataSetsMeta = pos->second;
  for (size_t idx= 0; idx < dataSetsMeta.size(); ++idx) {
    const DataSetMeta & dataSetMeta = dataSetsMeta.at(idx);
    if (dataSetMeta.name() == dsetName) {
      return dataSetMeta.dsetId();
    }
  }
  msg << "getDataSetId: groupId= " << groupId 
      << " " << hdf5util::objectName(groupId) 
      << " dsetName= " << dsetName
      << " was not found in datasets for group.";
  throw GroupMapException(ERR_LOC, msg.str());
}

// returns the positional index for a new dataset for the given group.
// Creates new entries in the maps if need be.  Throws an exception if 
// the dataset name already exists for the group.  Only to be called 
// when first creating a dataset in the group.
size_t HdfWriterGeneric::createNewDatasetSlotForGroup(hid_t groupId, const string & dsetName) 
{
  size_t dsetIndex = 0;
  std::map<hid_t, vector<DataSetMeta> >::iterator pos = m_datasetMap.find(groupId);
  if (pos == m_datasetMap.end()) {
    MsgLog(logger(), debug, "HdfWriterGeneric (" << m_debugName << ") adding groupId="
           << groupId << " " << hdf5util::objectName(groupId) << " to map.  Creating first slot for dataset " << dsetName);
    m_datasetMap[groupId]=vector<DataSetMeta>(1);
    return 0;
  }
  vector<DataSetMeta> &dsetMetaList = pos->second;
  for (size_t idx = 0; idx < dsetMetaList.size(); ++idx) {
    if (dsetMetaList[idx].name() == dsetName) {
      ostringstream msg;
      msg <<  "dsetname = " << dsetName 
          << " already created for group id " << groupId
          << " " << hdf5util::objectName(groupId)
          << " at idx=" << idx << " writer( " << m_debugName <<" )";
      throw WriteException(ERR_LOC, msg.str());
    }
  }
  dsetIndex = dsetMetaList.size();
  dsetMetaList.resize(dsetIndex+1);
  return dsetIndex;
}
