#include "Translator/EpicsWriteBuffer.h"
#include <string.h>
#include <map>

void EpicsWriteBufferDetails::copyValueFld(Translator::Unroll::EpicsPvCtrlString::valueBaseType &dest, const char * src) 
{
  strncpy(dest, src, Psana::Epics::MAX_STRING_SIZE);
}

template <>
int EpicsWriteBufferDetails::getNumberOfStringsForCtrlEnum<Translator::Unroll::EpicsPvCtrlEnum>(const Psana::Epics::EpicsPvCtrlEnum &psanaSrc) 
{
  return psanaSrc.dbr().no_str();
}

using namespace Translator;

namespace {

  const char * logger = "EpicsWriterBuffer";

  hid_t getH5TypeId_epicsTimeStamp() {
    static hid_t typeId = -1;
    if (typeId == -1) {
      typeId = H5Tcreate(H5T_COMPOUND, sizeof(Unroll::epicsTimeStamp));
      if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for epicsTimeStamp");
      herr_t status = 0;
      status = std::min(status, H5Tinsert(typeId, "secPastEpoch", offsetof(Unroll::epicsTimeStamp, secPastEpoch), H5T_NATIVE_INT32));
      status = std::min(status, H5Tinsert(typeId, "nsec", offsetof(Unroll::epicsTimeStamp, nsec), H5T_NATIVE_INT32));
      if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for epicsTimeStamp"); 
    }
    return typeId;
  }  

  hid_t createH5TypeId_EpicsPvCtrlString(hid_t pvNameType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvCtrlString::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvCtrlString) + ((numElem)-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvCtrlString");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvCtrlString, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvCtrlString, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvCtrlString, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "pvname", offsetof(Unroll::EpicsPvCtrlString, sPvName), pvNameType));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvCtrlString, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvCtrlString, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvCtrlString, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvCtrlString"); 
    return typeId;
  }

  hid_t createH5TypeId_EpicsPvCtrlShort(hid_t pvNameType, hid_t unitsType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvCtrlString::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvCtrlShort) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvCtrlShort");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvCtrlShort, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvCtrlShort, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvCtrlShort, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "pvname", offsetof(Unroll::EpicsPvCtrlShort, sPvName), pvNameType));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvCtrlShort, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvCtrlShort, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "units", offsetof(Unroll::EpicsPvCtrlShort, units), unitsType));
    status = std::min(status, H5Tinsert(typeId, "upper_disp_limit", offsetof(Unroll::EpicsPvCtrlShort, upper_disp_limit), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "lower_disp_limit", offsetof(Unroll::EpicsPvCtrlShort, lower_disp_limit), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "upper_alarm_limit", offsetof(Unroll::EpicsPvCtrlShort, upper_alarm_limit), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "upper_warning_limit", offsetof(Unroll::EpicsPvCtrlShort, upper_warning_limit), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "lower_warning_limit", offsetof(Unroll::EpicsPvCtrlShort, lower_warning_limit), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "lower_alarm_limit", offsetof(Unroll::EpicsPvCtrlShort, lower_alarm_limit), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "upper_ctrl_limit", offsetof(Unroll::EpicsPvCtrlShort, upper_ctrl_limit), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "lower_ctrl_limit", offsetof(Unroll::EpicsPvCtrlShort, lower_ctrl_limit), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvCtrlShort, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvCtrlShort"); 
    return typeId;
  }

  hid_t createH5TypeId_EpicsPvCtrlFloat(hid_t pvNameType, hid_t unitsType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvCtrlFloat::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvCtrlFloat) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvCtrlFloat");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvCtrlFloat, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvCtrlFloat, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvCtrlFloat, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "pvname", offsetof(Unroll::EpicsPvCtrlFloat, sPvName), pvNameType));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvCtrlFloat, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvCtrlFloat, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "precision", offsetof(Unroll::EpicsPvCtrlFloat, precision), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "units", offsetof(Unroll::EpicsPvCtrlFloat, units), unitsType));
    status = std::min(status, H5Tinsert(typeId, "upper_disp_limit", offsetof(Unroll::EpicsPvCtrlFloat, upper_disp_limit), H5T_NATIVE_FLOAT));
    status = std::min(status, H5Tinsert(typeId, "lower_disp_limit", offsetof(Unroll::EpicsPvCtrlFloat, lower_disp_limit), H5T_NATIVE_FLOAT));
    status = std::min(status, H5Tinsert(typeId, "upper_alarm_limit", offsetof(Unroll::EpicsPvCtrlFloat, upper_alarm_limit), H5T_NATIVE_FLOAT));
    status = std::min(status, H5Tinsert(typeId, "upper_warning_limit", offsetof(Unroll::EpicsPvCtrlFloat, upper_warning_limit), H5T_NATIVE_FLOAT));
    status = std::min(status, H5Tinsert(typeId, "lower_warning_limit", offsetof(Unroll::EpicsPvCtrlFloat, lower_warning_limit), H5T_NATIVE_FLOAT));
    status = std::min(status, H5Tinsert(typeId, "lower_alarm_limit", offsetof(Unroll::EpicsPvCtrlFloat, lower_alarm_limit), H5T_NATIVE_FLOAT));
    status = std::min(status, H5Tinsert(typeId, "upper_ctrl_limit", offsetof(Unroll::EpicsPvCtrlFloat, upper_ctrl_limit), H5T_NATIVE_FLOAT));
    status = std::min(status, H5Tinsert(typeId, "lower_ctrl_limit", offsetof(Unroll::EpicsPvCtrlFloat, lower_ctrl_limit), H5T_NATIVE_FLOAT));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvCtrlFloat, value),valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvCtrlFloat"); 
    return typeId;
  }
  
  hid_t createH5TypeId_EpicsPvCtrlEnum(hid_t pvNameType, hid_t strsArrayType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvCtrlEnum::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvCtrlEnum) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvCtrlEnum");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvCtrlEnum, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvCtrlEnum, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvCtrlEnum, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "pvname", offsetof(Unroll::EpicsPvCtrlEnum, sPvName), pvNameType));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvCtrlEnum, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvCtrlEnum, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "no_str", offsetof(Unroll::EpicsPvCtrlEnum, no_str), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "strs", offsetof(Unroll::EpicsPvCtrlEnum, strs), strsArrayType));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvCtrlEnum, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvCtrlEnum"); 
    return typeId;
  }
  
  hid_t createH5TypeId_EpicsPvCtrlChar(hid_t pvNameType, hid_t unitsType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvCtrlChar::valueBaseType);
    
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvCtrlChar) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvCtrlChar");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvCtrlChar, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvCtrlChar, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvCtrlChar, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "pvname", offsetof(Unroll::EpicsPvCtrlChar, sPvName), pvNameType));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvCtrlChar, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvCtrlChar, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "units", offsetof(Unroll::EpicsPvCtrlChar, units), unitsType));
    status = std::min(status, H5Tinsert(typeId, "upper_disp_limit", offsetof(Unroll::EpicsPvCtrlChar, upper_disp_limit), H5T_NATIVE_UINT8));
    status = std::min(status, H5Tinsert(typeId, "lower_disp_limit", offsetof(Unroll::EpicsPvCtrlChar, lower_disp_limit), H5T_NATIVE_UINT8));
    status = std::min(status, H5Tinsert(typeId, "upper_alarm_limit", offsetof(Unroll::EpicsPvCtrlChar, upper_alarm_limit), H5T_NATIVE_UINT8));
    status = std::min(status, H5Tinsert(typeId, "upper_warning_limit", offsetof(Unroll::EpicsPvCtrlChar, upper_warning_limit), H5T_NATIVE_UINT8));
    status = std::min(status, H5Tinsert(typeId, "lower_warning_limit", offsetof(Unroll::EpicsPvCtrlChar, lower_warning_limit), H5T_NATIVE_UINT8));
    status = std::min(status, H5Tinsert(typeId, "lower_alarm_limit", offsetof(Unroll::EpicsPvCtrlChar, lower_alarm_limit), H5T_NATIVE_UINT8));
    status = std::min(status, H5Tinsert(typeId, "upper_ctrl_limit", offsetof(Unroll::EpicsPvCtrlChar, upper_ctrl_limit), H5T_NATIVE_UINT8));
    status = std::min(status, H5Tinsert(typeId, "lower_ctrl_limit", offsetof(Unroll::EpicsPvCtrlChar, lower_ctrl_limit), H5T_NATIVE_UINT8));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvCtrlChar, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvCtrlChar"); 
    return typeId;
  }

  hid_t createH5TypeId_EpicsPvCtrlLong(hid_t pvNameType, hid_t unitsType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvCtrlLong::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvCtrlLong) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvCtrlLong");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvCtrlLong, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvCtrlLong, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvCtrlLong, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "pvname", offsetof(Unroll::EpicsPvCtrlLong, sPvName), pvNameType));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvCtrlLong, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvCtrlLong, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "units", offsetof(Unroll::EpicsPvCtrlLong, units), unitsType));
    status = std::min(status, H5Tinsert(typeId, "upper_disp_limit", offsetof(Unroll::EpicsPvCtrlLong, upper_disp_limit), H5T_NATIVE_INT32));
    status = std::min(status, H5Tinsert(typeId, "lower_disp_limit", offsetof(Unroll::EpicsPvCtrlLong, lower_disp_limit), H5T_NATIVE_INT32));
    status = std::min(status, H5Tinsert(typeId, "upper_alarm_limit", offsetof(Unroll::EpicsPvCtrlLong, upper_alarm_limit), H5T_NATIVE_INT32));
    status = std::min(status, H5Tinsert(typeId, "upper_warning_limit", offsetof(Unroll::EpicsPvCtrlLong, upper_warning_limit), H5T_NATIVE_INT32));
    status = std::min(status, H5Tinsert(typeId, "lower_warning_limit", offsetof(Unroll::EpicsPvCtrlLong, lower_warning_limit), H5T_NATIVE_INT32));
    status = std::min(status, H5Tinsert(typeId, "lower_alarm_limit", offsetof(Unroll::EpicsPvCtrlLong, lower_alarm_limit), H5T_NATIVE_INT32));
    status = std::min(status, H5Tinsert(typeId, "upper_ctrl_limit", offsetof(Unroll::EpicsPvCtrlLong, upper_ctrl_limit), H5T_NATIVE_INT32));
    status = std::min(status, H5Tinsert(typeId, "lower_ctrl_limit", offsetof(Unroll::EpicsPvCtrlLong, lower_ctrl_limit), H5T_NATIVE_INT32));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvCtrlLong, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvCtrlLong"); 
    return typeId;
  }

  hid_t createH5TypeId_EpicsPvCtrlDouble(hid_t pvNameType, hid_t unitsType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvCtrlDouble::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvCtrlDouble) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvCtrlDouble");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvCtrlDouble, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvCtrlDouble, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvCtrlDouble, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "pvname", offsetof(Unroll::EpicsPvCtrlDouble, sPvName), pvNameType));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvCtrlDouble, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvCtrlDouble, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "precision", offsetof(Unroll::EpicsPvCtrlDouble, precision), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "units", offsetof(Unroll::EpicsPvCtrlDouble, units), unitsType));
    status = std::min(status, H5Tinsert(typeId, "upper_disp_limit", offsetof(Unroll::EpicsPvCtrlDouble, upper_disp_limit), H5T_NATIVE_DOUBLE));
    status = std::min(status, H5Tinsert(typeId, "lower_disp_limit", offsetof(Unroll::EpicsPvCtrlDouble, lower_disp_limit), H5T_NATIVE_DOUBLE));
    status = std::min(status, H5Tinsert(typeId, "upper_alarm_limit", offsetof(Unroll::EpicsPvCtrlDouble, upper_alarm_limit), H5T_NATIVE_DOUBLE));
    status = std::min(status, H5Tinsert(typeId, "upper_warning_limit", offsetof(Unroll::EpicsPvCtrlDouble, upper_warning_limit), H5T_NATIVE_DOUBLE));
    status = std::min(status, H5Tinsert(typeId, "lower_warning_limit", offsetof(Unroll::EpicsPvCtrlDouble, lower_warning_limit), H5T_NATIVE_DOUBLE));
    status = std::min(status, H5Tinsert(typeId, "lower_alarm_limit", offsetof(Unroll::EpicsPvCtrlDouble, lower_alarm_limit), H5T_NATIVE_DOUBLE));
    status = std::min(status, H5Tinsert(typeId, "upper_ctrl_limit", offsetof(Unroll::EpicsPvCtrlDouble, upper_ctrl_limit), H5T_NATIVE_DOUBLE));
    status = std::min(status, H5Tinsert(typeId, "lower_ctrl_limit", offsetof(Unroll::EpicsPvCtrlDouble, lower_ctrl_limit), H5T_NATIVE_DOUBLE));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvCtrlDouble, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvCtrlDouble"); 
    return typeId;
  }

  hid_t createH5TypeId_EpicsPvTimeString(hid_t stringType, hid_t stampType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvTimeString::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvTimeString) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvTimeString");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvTimeString, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvTimeString, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvTimeString, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvTimeString, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvTimeString, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "stamp", offsetof(Unroll::EpicsPvTimeString, stamp), stampType));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvTimeString, value), stringType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvTimeString"); 
    return typeId;
  }

  hid_t createH5TypeId_EpicsPvTimeShort(hid_t stampType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvTimeShort::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, sizeof(Unroll::EpicsPvTimeShort) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvTimeShort");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvTimeShort, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvTimeShort, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvTimeShort, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvTimeShort, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvTimeShort, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "stamp", offsetof(Unroll::EpicsPvTimeShort, stamp), stampType));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvTimeShort, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvTimeShort"); 
    return typeId;
  }

  hid_t createH5TypeId_EpicsPvTimeFloat(hid_t stampType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvTimeFloat::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvTimeFloat) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvTimeFloat");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvTimeFloat, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvTimeFloat, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvTimeFloat, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvTimeFloat, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvTimeFloat, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "stamp", offsetof(Unroll::EpicsPvTimeFloat, stamp), stampType));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvTimeFloat, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvTimeFloat"); 
    return typeId;
  }

  hid_t createH5TypeId_EpicsPvTimeEnum(hid_t stampType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvTimeEnum::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvTimeEnum) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvTimeEnum");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvTimeEnum, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvTimeEnum, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvTimeEnum, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvTimeEnum, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvTimeEnum, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "stamp", offsetof(Unroll::EpicsPvTimeEnum, stamp), stampType));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvTimeEnum, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvTimeEnum"); 
    return typeId;
  }

  hid_t createH5TypeId_EpicsPvTimeChar(hid_t stampType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvTimeChar::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvTimeChar) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvTimeChar");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvTimeChar, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvTimeChar, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvTimeChar, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvTimeChar, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvTimeChar, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "stamp", offsetof(Unroll::EpicsPvTimeChar, stamp), stampType));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvTimeChar, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvTimeChar"); 
    return typeId;
  }

  hid_t createH5TypeId_EpicsPvTimeLong(hid_t stampType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvTimeLong::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvTimeLong) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvTimeLong");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvTimeLong, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvTimeLong, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvTimeLong, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvTimeLong, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvTimeLong, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "stamp", offsetof(Unroll::EpicsPvTimeLong, stamp), stampType));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvTimeLong, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvTimeLong"); 
    return typeId;
  }
  
  hid_t createH5TypeId_EpicsPvTimeDouble(hid_t stampType, hid_t valueType, int numElem) {
    size_t valueBaseSize = sizeof(Unroll::EpicsPvTimeDouble::valueBaseType);
    hid_t typeId = H5Tcreate(H5T_COMPOUND, 
                             sizeof(Unroll::EpicsPvTimeDouble) + (numElem-1)*valueBaseSize);
    if (typeId < 0) MsgLog(logger, fatal, "Failed to create h5 type id for EpicsPvTimeDouble");
    herr_t status = 0;
    status = std::min(status, H5Tinsert(typeId, "pvId", offsetof(Unroll::EpicsPvTimeDouble, iPvId), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "dbrType", offsetof(Unroll::EpicsPvTimeDouble, iDbrType), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "numElements", offsetof(Unroll::EpicsPvTimeDouble, iNumElements), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "status", offsetof(Unroll::EpicsPvTimeDouble, status), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "severity", offsetof(Unroll::EpicsPvTimeDouble, severity), H5T_NATIVE_INT16));
    status = std::min(status, H5Tinsert(typeId, "stamp", offsetof(Unroll::EpicsPvTimeDouble, stamp), stampType));
    status = std::min(status, H5Tinsert(typeId, "value", offsetof(Unroll::EpicsPvTimeDouble, value), valueType));
    if (status < 0) MsgLog(logger, fatal, "error inserting field into h5 typeId for EpicsPvTimeDouble"); 
    return typeId;
  }


  class EpicsTypeManager {
  public:
    EpicsTypeManager() :
      m_enumStrsArrayTypes(Psana::Epics::MAX_ENUM_STATES+1)
    {
      for (unsigned idx=0; idx < m_enumStrsArrayTypes.size(); ++idx) {
        m_enumStrsArrayTypes.at(idx) = -1;
      }
      makeSharedTypes();
    };
    
    ~EpicsTypeManager() {
      closeSharedTypes();
    };
    
    hid_t memH5Type(int16_t dbrType, int numElem, int numStrsCtrlEnum) {
      return h5Type(dbrType, numElem, numStrsCtrlEnum, true);
    };
    
    hid_t fileH5Type(int16_t dbrType, int numElem, int numStrsCtrlEnum) {
      return h5Type(dbrType, numElem, numStrsCtrlEnum, false);
    };
    
  protected:
    const char * logger() { return "EpicsTypeManager"; }

    enum ValueBase {epicsUnknown, epicsString, epicsShort, epicsFloat, 
                    epicsEnum, epicsChar, epicsLong, epicsDouble};

    hid_t getValueType(int16_t dbrType, int numElem) { 
      hid_t baseType = -1;
      enum ValueBase valueBaseKey = epicsUnknown;
      switch (dbrType) {
      case Psana::Epics::DBR_CTRL_STRING:
      case Psana::Epics::DBR_TIME_STRING:
        baseType = m_stringType;
        valueBaseKey = epicsString;
        break;
      case Psana::Epics::DBR_CTRL_SHORT:
      case Psana::Epics::DBR_TIME_SHORT:
        baseType = H5T_NATIVE_INT16;
        valueBaseKey = epicsShort;
        break;
      case Psana::Epics::DBR_CTRL_FLOAT:
      case Psana::Epics::DBR_TIME_FLOAT:
        baseType = H5T_NATIVE_FLOAT;
        valueBaseKey = epicsFloat;
        break;
      case Psana::Epics::DBR_CTRL_ENUM:
      case Psana::Epics::DBR_TIME_ENUM:
        baseType = H5T_NATIVE_UINT16;
        valueBaseKey = epicsEnum;
        break;
      case Psana::Epics::DBR_CTRL_CHAR:
      case Psana::Epics::DBR_TIME_CHAR:
        baseType = H5T_NATIVE_UINT8;
        valueBaseKey = epicsChar;
        break;
      case Psana::Epics::DBR_CTRL_LONG:
      case Psana::Epics::DBR_TIME_LONG:
        baseType = H5T_NATIVE_INT32;
        valueBaseKey = epicsLong;
        break;
      case Psana::Epics::DBR_CTRL_DOUBLE:
      case Psana::Epics::DBR_TIME_DOUBLE:
        baseType = H5T_NATIVE_DOUBLE;
        valueBaseKey = epicsDouble;
        break;
      default:
        MsgLog(logger(), warning, 
               "unexpected dbr type: " << dbrType 
               << " in epics h5 value type");
        return -1;
      } // switch (dbrType)
      if (numElem == 1) return baseType;
      ValueArrayTypeKey valueArrayKey(valueBaseKey, numElem);
      std::map< ValueArrayTypeKey, hid_t>::iterator pos = m_valueArrayTypes.find(valueArrayKey);
      if (pos != m_valueArrayTypes.end()) return pos->second;
      const unsigned rank = 1;
      hsize_t dims[rank] = {numElem};
      hid_t valueArrayType = H5Tarray_create2(baseType, rank, dims);
      if (valueArrayType < 0) {
        MsgLog(logger(), error, "H5Tarray_create2 call failed: baseType="
               << baseType << " valueBaseKey=" << valueBaseKey
               << " numElem: " << numElem);
        return -1; 
      }
      m_valueArrayTypes[valueArrayKey]=valueArrayType;
      return valueArrayType;
    }
    
    hid_t h5Type(int16_t dbrType, int numElem, int numStrsCtrlEnum, bool mem) {
      TypeKey typeKey(dbrType, numElem, numStrsCtrlEnum);
      std::map<TypeKey, hid_t>::iterator memPos = m_h5memTypes.find(typeKey);
      if (memPos != m_h5memTypes.end()) {
        if ((not mem) and (dbrType == Psana::Epics::DBR_CTRL_ENUM)) {
          std::map<TypeKey, hid_t>::iterator filePos = m_h5fileTypes.find(typeKey);
          if (filePos == m_h5fileTypes.end()) {
            MsgLog(logger(), fatal, "epics h5 mem type is set for ctrl enum, BUT file type is NOT");
          }
          return filePos->second;
        }
        return memPos->second;
      }
      // need to build, cache, and return type
      hid_t valueType = getValueType(dbrType, numElem);
      hid_t memTypeId = -1;
      hid_t strsType = -1;
      const unsigned rank = 1;
      hid_t stampType = getH5TypeId_epicsTimeStamp();
      hsize_t dims[rank] = {numStrsCtrlEnum};

      switch (dbrType) {
      case Psana::Epics::DBR_CTRL_STRING:
        memTypeId = createH5TypeId_EpicsPvCtrlString(m_pvNameType, valueType, numElem);
        break;
      case Psana::Epics::DBR_CTRL_SHORT:
        memTypeId = createH5TypeId_EpicsPvCtrlShort(m_pvNameType, m_unitsType, valueType, numElem);
        break;
      case Psana::Epics::DBR_CTRL_FLOAT:
        memTypeId = createH5TypeId_EpicsPvCtrlFloat(m_pvNameType, m_unitsType, valueType, numElem);
        break;
      case Psana::Epics::DBR_CTRL_ENUM:
        if (numStrsCtrlEnum < 0) {
          MsgLog(logger(),fatal,
                 "get h5Type for DBR_CTRL_ENUM called with negative number of strs");
        }
        if (numStrsCtrlEnum > Psana::Epics::MAX_ENUM_STATES) {
          MsgLog(logger(),warning,
                 "setting number of strs for DBR_CTRL_ENUM  to max. It was: " << numStrsCtrlEnum);
          numStrsCtrlEnum = Psana::Epics::MAX_ENUM_STATES;
        }
        strsType = m_enumStrsArrayTypes.at(numStrsCtrlEnum);
        if (strsType == -1) {
          strsType = H5Tarray_create2(m_enumStrType, rank, dims);
          if (strsType < 0) {
            MsgLog(logger(), fatal, 
                   "failed to create array type for enum strings 'strs' field with" << numStrsCtrlEnum << " items");
          }
          m_enumStrsArrayTypes.at(numStrsCtrlEnum) = strsType;
          MsgLog(logger(),debug,"created ctrl enum array string for " << numStrsCtrlEnum << " strings, hid=" << strsType);
        }
        memTypeId = createH5TypeId_EpicsPvCtrlEnum(m_pvNameType, strsType, valueType, numElem);
        MsgLog(logger(),debug,"created ctrlenum type for " << numStrsCtrlEnum << " strings, hid=" << memTypeId);
        break;
      case Psana::Epics::DBR_CTRL_CHAR:
        memTypeId = createH5TypeId_EpicsPvCtrlChar(m_pvNameType, m_unitsType, valueType, numElem);
        break;
      case Psana::Epics::DBR_CTRL_LONG:
        memTypeId = createH5TypeId_EpicsPvCtrlLong(m_pvNameType, m_unitsType, valueType, numElem);
        break;
      case Psana::Epics::DBR_CTRL_DOUBLE:
        memTypeId = createH5TypeId_EpicsPvCtrlDouble(m_pvNameType, m_unitsType, valueType, numElem);
        break;
      case Psana::Epics::DBR_TIME_STRING:
        memTypeId = createH5TypeId_EpicsPvTimeString(m_stringType, stampType, valueType, numElem);
        break;
      case Psana::Epics::DBR_TIME_SHORT:
        memTypeId = createH5TypeId_EpicsPvTimeShort(stampType, valueType, numElem);
        break;
      case Psana::Epics::DBR_TIME_FLOAT:
        memTypeId = createH5TypeId_EpicsPvTimeFloat(stampType, valueType, numElem);
        break;
      case Psana::Epics::DBR_TIME_ENUM:
        memTypeId = createH5TypeId_EpicsPvTimeEnum(stampType, valueType, numElem);
        break;
      case Psana::Epics::DBR_TIME_CHAR:
        memTypeId = createH5TypeId_EpicsPvTimeChar(stampType, valueType, numElem);
        break;
      case Psana::Epics::DBR_TIME_LONG:
        memTypeId = createH5TypeId_EpicsPvTimeLong(stampType, valueType, numElem);
        break;
      case Psana::Epics::DBR_TIME_DOUBLE:
        memTypeId = createH5TypeId_EpicsPvTimeDouble(stampType, valueType, numElem);
        break;
      default:
        MsgLog(logger(), warning, "unexpected dbr type: " << dbrType << " in epics h5 type");
      } // switch (dbrType)

      m_h5memTypes[typeKey] = memTypeId;

      if ((not mem) and (dbrType == Psana::Epics::DBR_CTRL_ENUM)) {
        hid_t fileTypeId = H5Tcopy(memTypeId);
        herr_t status = H5Tpack(fileTypeId);
        if (fileTypeId < 0 or status < 0) MsgLog(logger(), fatal, "H5Tcopy or H5Tpack failed");
        m_h5fileTypes[typeKey] = fileTypeId;
        return fileTypeId;
      }
      return memTypeId;
    };
    
    void makeSharedTypes() {
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
    }
    
    void closeSharedTypes() {
      herr_t status = H5Tclose(m_pvNameType);
      status = std::min(status, H5Tclose(m_stringType));
      status = std::min(status, H5Tclose(m_unitsType));
      status = std::min(status, H5Tclose(m_enumStrType));
      
      for (unsigned idx=0; idx < m_enumStrsArrayTypes.size(); ++idx) {
        if (m_enumStrsArrayTypes.at(idx) !=-1) {
          status = std::min(status, H5Tclose(m_enumStrsArrayTypes.at(idx)));
          m_enumStrsArrayTypes.at(idx) = -1;
        }
      }
      if (status<0) MsgLog(logger(), fatal, "failed to close one of the shared epics pv datatypes");
      
      std::map<TypeKey, hid_t>::iterator pos;
      for (pos = m_h5memTypes.begin(); pos != m_h5memTypes.end(); ++pos) {
        if (pos->second != -1) {
          herr_t status = H5Tclose(pos->second);
          if (status<0) MsgLog(logger(), error, "failed to close h5 type for mem epics pv for dbr=" 
                               << pos->first.dbrType << " numElem=" 
                               << pos->first.numElem << " numStrsCtrlEnumm="
                               << pos->first.numStrsCtrlEnum);
        }
      }
      for (pos = m_h5fileTypes.begin(); pos != m_h5fileTypes.end(); ++pos) {
        if (pos->second != -1) {
          herr_t status = H5Tclose(pos->second);
          if (status<0) MsgLog(logger(), error, "failed to close h5 type for file epics pv for dbr=" 
                               << pos->first.dbrType << " numElem=" 
                               << pos->first.numElem << " numStrsCtrlEnumm="
                               << pos->first.numStrsCtrlEnum);
        }
      }
      m_h5fileTypes.clear();
      m_h5memTypes.clear();

      std::map<ValueArrayTypeKey,hid_t>::iterator valueArrayPos;
      for (valueArrayPos = m_valueArrayTypes.begin(); 
           valueArrayPos != m_valueArrayTypes.end(); ++valueArrayPos) {
        if (H5Tclose(valueArrayPos->second) < 0) {
          MsgLog(logger(), error, "failed to close value array type. "
                 << "valueBaseKey=" << (valueArrayPos->first).first
                 << " numElem=" << (valueArrayPos->first).second);
        }
      }
      m_valueArrayTypes.clear();
    }
  private:
    std::vector<hid_t> m_enumStrsArrayTypes;
    typedef std::pair<enum ValueBase, int> ValueArrayTypeKey;
    std::map< ValueArrayTypeKey, hid_t> m_valueArrayTypes;

    struct TypeKey {
      uint16_t dbrType;
      int numElem;
      int numStrsCtrlEnum;
      TypeKey() : dbrType(0), numElem(-1), numStrsCtrlEnum(-1) {};
      TypeKey(uint16_t _dbrType, int _numElem, int _numStrsCtrlEnum) :
        dbrType(_dbrType), numElem(_numElem), numStrsCtrlEnum(_numStrsCtrlEnum) {};
      TypeKey(const TypeKey &o) :
        dbrType(o.dbrType), numElem(o.numElem), numStrsCtrlEnum(o.numStrsCtrlEnum) {};
      TypeKey &operator=(const TypeKey &o) {
        dbrType = o.dbrType;
        numElem = o.numElem;
        numStrsCtrlEnum = o.numStrsCtrlEnum;
        return *this;
      }
      
      bool operator <(const TypeKey &other) const {
        if (dbrType < other.dbrType) return true;
        if (dbrType > other.dbrType) return false;
        if (numElem < other.numElem) return true;
        if (numElem > other.numElem) return false;
        if (numStrsCtrlEnum < other.numStrsCtrlEnum) return true;
        if (numStrsCtrlEnum > other.numStrsCtrlEnum) return false;
        return false; // everything is equal
      }
    };
    
    std::map<TypeKey, hid_t>  m_h5memTypes, m_h5fileTypes;   
    
    // base h5 types that are used in the Epics Pv types
    hid_t m_pvNameType;  // char[Psana::Epics::iMaxPvNameLength
    hid_t m_stringType;  // char[Psana::Epics:: MAX_STRING_SIZE]
    hid_t m_unitsType;   // char[Psana::Epics::MAX_UNITS_SIZE]
    hid_t m_enumStrType; // char[Psana::Epics::MAX_ENUM_STRING_SIZE]
  }; // class EpicsTypeManager
  
  EpicsTypeManager epicsTypeManager;
  
} // local namespace

hid_t EpicsWriteBufferDetails::epicsMemH5Type(int16_t dbrType, int numElem, int numStrsCtrlEnum)
{
  return epicsTypeManager.memH5Type(dbrType, numElem, numStrsCtrlEnum);
}

hid_t EpicsWriteBufferDetails::epicsFileH5Type(int16_t dbrType, int numElem, int numStrsCtrlEnum)
{
  return epicsTypeManager.fileH5Type(dbrType, numElem, numStrsCtrlEnum);
}

