#include <string>

#include "boost/make_shared.hpp"

#include "MsgLogger/MsgLogger.h"

#include "Translator/HdfWriterEpicsPv.h"
#include "Translator/epics.ddl.h"

using namespace Translator;
using namespace std;

namespace {

const string logger(const string addTo="") { 
  static const string base("Translator.HdfWriterEpicsPv");
  if (addTo.size()>0) {
    return base + std::string(".") + addTo;
  }
  return base;
}

} // namespace

std::ostream & Translator::operator<<(std::ostream & o, HdfWriterEpicsPv::DispatchAction da) {
  if (da==HdfWriterEpicsPv::CreateWriteClose) o << "CreateWriteClose";
  else if (da==HdfWriterEpicsPv::CreateAppend) o << "CreateAppend";
  else if (da ==HdfWriterEpicsPv::Append) o << "Append";
  else o << "**unknown**";
  return o;
}

HdfWriterEpicsPv::HdfWriterEpicsPv(const DataSetCreationProperties & dataSetCreationProperties,
                                   boost::shared_ptr<HdfWriterEventId> hdfWriterEventId) 
  : m_dataSetCreationProperties(dataSetCreationProperties),
    m_hdfWriterEventId(hdfWriterEventId),
    m_enumStrsArrayTypes(Psana::Epics::MAX_ENUM_STATES+1),
    m_numberStringsToH5TypeForCtrlEnum(Psana::Epics::MAX_ENUM_STATES+1)
{
  m_hdfWriterGeneric = boost::make_shared<HdfWriterGeneric>("epics");
  for (unsigned idx=0; idx < m_enumStrsArrayTypes.size(); ++idx) {
    m_enumStrsArrayTypes.at(idx) = -1;
    m_numberStringsToH5TypeForCtrlEnum.at(idx) = -1;
  }
  makeSharedTypes();
}

void HdfWriterEpicsPv::makeSharedTypes() {
  m_pvNameType = H5Tcopy(H5T_C_S1);
  herr_t status = H5Tset_size(m_pvNameType, Psana::Epics::iMaxPvNameLength);
  if (m_pvNameType<0 or status<0) MsgLog(logger(),fatal, "failed to create pvName type");

  m_stringType = H5Tcopy(H5T_C_S1);
  status = H5Tset_size(m_stringType, Psana::Epics:: MAX_STRING_SIZE);
  if (m_stringType<0 or status<0) MsgLog(logger(),fatal, "failed to create string type");

  m_unitsType = H5Tcopy(H5T_C_S1);
  status = H5Tset_size(m_unitsType, Psana::Epics::MAX_UNITS_SIZE);
  if (m_unitsType<0 or status<0) MsgLog(logger(),fatal, "failed to create units type");
  
  m_enumStrType = H5Tcopy(H5T_C_S1);
  status = H5Tset_size(m_enumStrType, Psana::Epics::MAX_ENUM_STRING_SIZE);
  if (m_enumStrType<0 or status<0) MsgLog(logger(),fatal, "failed to create enum type for one symbol");

  m_stampType = createH5TypeId_epicsTimeStamp();
  if (m_stampType<0) MsgLog(logger(),fatal, "failed to create enum time stamp hdf5 type id");
}

void HdfWriterEpicsPv::closeDataset(hid_t groupId) { 
  m_hdfWriterGeneric->closeDatasets(groupId); 
  m_hdfWriterEventId->closeDataset(groupId); 
}

void HdfWriterEpicsPv::closeDatasetsForAllGroups() { 
  m_hdfWriterGeneric->closeDatasetsForAllGroups();
  m_hdfWriterEventId->closeDatasetsForAllGroups(); 
}

void HdfWriterEpicsPv::closeSharedTypes() {
  herr_t status = H5Tclose(m_pvNameType);
  status = std::min(status, H5Tclose(m_stringType));
  status = std::min(status, H5Tclose(m_unitsType));
  //  status = std::min(status, H5Tclose(m_allEnumStrsType));
  status = std::min(status, H5Tclose(m_enumStrType));
  status = std::min(status, H5Tclose(m_stampType));

  for (unsigned idx=0; idx < m_enumStrsArrayTypes.size(); ++idx) {
    if (m_enumStrsArrayTypes.at(idx) !=-1) {
      status = std::min(status, H5Tclose(m_enumStrsArrayTypes.at(idx)));
      m_enumStrsArrayTypes.at(idx) = -1;
    }
    if (m_numberStringsToH5TypeForCtrlEnum.at(idx) !=-1) {
      status = std::min(status, H5Tclose(m_numberStringsToH5TypeForCtrlEnum.at(idx)));
      m_numberStringsToH5TypeForCtrlEnum.at(idx)=-1;
    }
  }
  if (status<0) MsgLog(logger(), fatal, "failed to close one of the shared epics pv datatypes");

  map<uint16_t, hid_t>::iterator pos;
  for (pos = m_dbr2h5TypeId.begin(); pos != m_dbr2h5TypeId.end(); ++pos) {
    herr_t status = H5Tclose(pos->second);
    if (status<0) MsgLog(logger(), fatal, "failed to close epics pv for dbr " << pos->first);
  }
  m_dbr2h5TypeId.clear();
}

HdfWriterEpicsPv::~HdfWriterEpicsPv() {
  closeSharedTypes();
}

hid_t HdfWriterEpicsPv::getCtrlEnumTypeId(int numberOfStrs) {  
  if (numberOfStrs<0) MsgLog(logger(),fatal,"getCtrlEnumTypeId called with -1 for strs");
  if (numberOfStrs > Psana::Epics::MAX_ENUM_STATES) {
    MsgLog(logger(),warning,"The no_str requested for a epics ctrl enum 'strs' field is " << numberOfStrs
           << " which is outside the range [" << 1 << ", " << Psana::Epics::MAX_ENUM_STATES
           << "] " << " h5 type will be created with " << Psana::Epics::MAX_ENUM_STATES 
           << " entries in 'strs'");
    numberOfStrs = Psana::Epics::MAX_ENUM_STATES;
  }
  hid_t ctrlEnumType = m_numberStringsToH5TypeForCtrlEnum.at(numberOfStrs);
  if (ctrlEnumType == -1) {
    hid_t strsType = m_enumStrsArrayTypes.at(numberOfStrs);
    if (strsType == -1) {
      const unsigned rank = 1;
      hsize_t dims[rank];
      dims[0] = numberOfStrs;
      strsType = H5Tarray_create2(m_enumStrType, rank, dims);
      if (strsType<0) {
        MsgLog(logger(), fatal, 
               "failed to create array type for enum strings 'strs' field with" << numberOfStrs << " items");
      }
      m_enumStrsArrayTypes.at(numberOfStrs) = strsType;
      MsgLog(logger(),debug,"created ctrl enum array string for " << numberOfStrs << " strings, hid=" << strsType);
    }
    ctrlEnumType = createH5TypeId_EpicsPvCtrlEnum(m_pvNameType, strsType, numberOfStrs);
    m_numberStringsToH5TypeForCtrlEnum.at(numberOfStrs) = ctrlEnumType;
    MsgLog(logger(),debug,"created ctrlenum type for " << numberOfStrs << " strings, hid=" << ctrlEnumType);
  }
  return ctrlEnumType;
}

hid_t HdfWriterEpicsPv::getTypeId(int16_t dbrType, int ctrlEnumNumberOfStrs) {  
  if (dbrType == Psana::Epics::DBR_CTRL_ENUM) return getCtrlEnumTypeId(ctrlEnumNumberOfStrs);

  map<uint16_t, hid_t>::iterator typeIdPos = m_dbr2h5TypeId.find(dbrType);
  if (typeIdPos != m_dbr2h5TypeId.end()) {
    return typeIdPos->second;
  }

  // it is not cached, build it
  hid_t typeId = -1;
  switch (dbrType) {
  case Psana::Epics::DBR_TIME_STRING:
    typeId = createH5TypeId_EpicsPvTimeString(m_stringType, m_stampType);
    break;
  case Psana::Epics::DBR_TIME_SHORT: // DBR_TIME_INT is the same as DBR_TIME_SHORT
    typeId = createH5TypeId_EpicsPvTimeShort(m_stampType);
    break;
  case Psana::Epics::DBR_TIME_FLOAT:
    typeId = createH5TypeId_EpicsPvTimeFloat(m_stampType);
    break;
  case Psana::Epics::DBR_TIME_ENUM:
    typeId = createH5TypeId_EpicsPvTimeEnum(m_stampType);
    break;
  case Psana::Epics::DBR_TIME_CHAR:
    typeId = createH5TypeId_EpicsPvTimeChar(m_stampType);
    break;
  case Psana::Epics::DBR_TIME_LONG:
    typeId = createH5TypeId_EpicsPvTimeLong(m_stampType);
    break;
  case Psana::Epics::DBR_TIME_DOUBLE:
    typeId = createH5TypeId_EpicsPvTimeDouble(m_stampType);
    break;
  case Psana::Epics::DBR_CTRL_STRING:
    typeId = createH5TypeId_EpicsPvCtrlString(m_pvNameType, m_stringType);
    break;
  case Psana::Epics::DBR_CTRL_SHORT: // DBR_CTRL_INT is the same as DBR_CTRL_SHORT
    typeId = createH5TypeId_EpicsPvCtrlShort(m_pvNameType, m_unitsType);
    break;
  case Psana::Epics::DBR_CTRL_FLOAT:
    typeId = createH5TypeId_EpicsPvCtrlFloat(m_pvNameType, m_unitsType);
    break;
  case Psana::Epics::DBR_CTRL_ENUM:
    // this case is handled above
    break;
  case Psana::Epics::DBR_CTRL_CHAR:
    typeId = createH5TypeId_EpicsPvCtrlChar(m_pvNameType, m_unitsType);
    break;
  case Psana::Epics::DBR_CTRL_LONG:
    typeId = createH5TypeId_EpicsPvCtrlLong(m_pvNameType,  m_unitsType);
    break;
  case Psana::Epics::DBR_CTRL_DOUBLE:
    typeId = createH5TypeId_EpicsPvCtrlDouble(m_pvNameType,  m_unitsType);
    break;
  default:
    MsgLog(logger(), fatal, "h5 type id for DBR type " << dbrType << " is not implemented");
  }
  
  m_dbr2h5TypeId[dbrType] = typeId;
  return typeId;
}


