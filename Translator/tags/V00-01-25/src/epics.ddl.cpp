/* Do not edit this file.  It is created by a code generator. */

#include <string.h>
#include <algorithm>
#include "MsgLogger/MsgLogger.h"
#include "Translator/epics.ddl.h"

namespace {

void copyEpicsPvCtrlEnumStrings(const Psana::Epics::EpicsPvCtrlEnum & sourceObject, 
                                Translator::Unroll::EpicsPvCtrlEnum & destObject)
{
  const Psana::Epics::dbr_ctrl_enum& dbr = sourceObject.dbr();
  for (uint16_t stringNumber = 0; stringNumber < dbr.no_str(); ++stringNumber) {
    strncpy(destObject.strs[stringNumber], dbr.strings(stringNumber), Psana::Epics::MAX_ENUM_STRING_SIZE);
  }
}

} // local namespace

namespace Translator {

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvCtrlString &source,
                    Unroll::EpicsPvCtrlString &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  strncpy(dest.sPvName, source.pvName(), Psana::Epics::iMaxPvNameLength);
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvCtrlShort &source,
                    Unroll::EpicsPvCtrlShort &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  strncpy(dest.sPvName, source.pvName(), Psana::Epics::iMaxPvNameLength);
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  strncpy(dest.units, source.dbr().units(), Psana::Epics::MAX_UNITS_SIZE);
  dest.upper_disp_limit = source.dbr().upper_disp_limit();
  dest.lower_disp_limit = source.dbr().lower_disp_limit();
  dest.upper_alarm_limit = source.dbr().upper_alarm_limit();
  dest.upper_warning_limit = source.dbr().upper_warning_limit();
  dest.lower_warning_limit = source.dbr().lower_warning_limit();
  dest.lower_alarm_limit = source.dbr().lower_alarm_limit();
  dest.upper_ctrl_limit = source.dbr().upper_ctrl_limit();
  dest.lower_ctrl_limit = source.dbr().lower_ctrl_limit();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvCtrlFloat &source,
                    Unroll::EpicsPvCtrlFloat &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  strncpy(dest.sPvName, source.pvName(), Psana::Epics::iMaxPvNameLength);
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  dest.precision = source.dbr().precision();
  strncpy(dest.units, source.dbr().units(), Psana::Epics::MAX_UNITS_SIZE);
  dest.upper_disp_limit = source.dbr().upper_disp_limit();
  dest.lower_disp_limit = source.dbr().lower_disp_limit();
  dest.upper_alarm_limit = source.dbr().upper_alarm_limit();
  dest.upper_warning_limit = source.dbr().upper_warning_limit();
  dest.lower_warning_limit = source.dbr().lower_warning_limit();
  dest.lower_alarm_limit = source.dbr().lower_alarm_limit();
  dest.upper_ctrl_limit = source.dbr().upper_ctrl_limit();
  dest.lower_ctrl_limit = source.dbr().lower_ctrl_limit();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvCtrlEnum &source,
                    Unroll::EpicsPvCtrlEnum &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  strncpy(dest.sPvName, source.pvName(), Psana::Epics::iMaxPvNameLength);
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  dest.no_str = source.dbr().no_str();
  copyEpicsPvCtrlEnumStrings(source, dest);
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvCtrlChar &source,
                    Unroll::EpicsPvCtrlChar &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  strncpy(dest.sPvName, source.pvName(), Psana::Epics::iMaxPvNameLength);
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  strncpy(dest.units, source.dbr().units(), Psana::Epics::MAX_UNITS_SIZE);
  dest.upper_disp_limit = source.dbr().upper_disp_limit();
  dest.lower_disp_limit = source.dbr().lower_disp_limit();
  dest.upper_alarm_limit = source.dbr().upper_alarm_limit();
  dest.upper_warning_limit = source.dbr().upper_warning_limit();
  dest.lower_warning_limit = source.dbr().lower_warning_limit();
  dest.lower_alarm_limit = source.dbr().lower_alarm_limit();
  dest.upper_ctrl_limit = source.dbr().upper_ctrl_limit();
  dest.lower_ctrl_limit = source.dbr().lower_ctrl_limit();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvCtrlLong &source,
                    Unroll::EpicsPvCtrlLong &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  strncpy(dest.sPvName, source.pvName(), Psana::Epics::iMaxPvNameLength);
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  strncpy(dest.units, source.dbr().units(), Psana::Epics::MAX_UNITS_SIZE);
  dest.upper_disp_limit = source.dbr().upper_disp_limit();
  dest.lower_disp_limit = source.dbr().lower_disp_limit();
  dest.upper_alarm_limit = source.dbr().upper_alarm_limit();
  dest.upper_warning_limit = source.dbr().upper_warning_limit();
  dest.lower_warning_limit = source.dbr().lower_warning_limit();
  dest.lower_alarm_limit = source.dbr().lower_alarm_limit();
  dest.upper_ctrl_limit = source.dbr().upper_ctrl_limit();
  dest.lower_ctrl_limit = source.dbr().lower_ctrl_limit();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvCtrlDouble &source,
                    Unroll::EpicsPvCtrlDouble &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  strncpy(dest.sPvName, source.pvName(), Psana::Epics::iMaxPvNameLength);
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  dest.precision = source.dbr().precision();
  strncpy(dest.units, source.dbr().units(), Psana::Epics::MAX_UNITS_SIZE);
  dest.upper_disp_limit = source.dbr().upper_disp_limit();
  dest.lower_disp_limit = source.dbr().lower_disp_limit();
  dest.upper_alarm_limit = source.dbr().upper_alarm_limit();
  dest.upper_warning_limit = source.dbr().upper_warning_limit();
  dest.lower_warning_limit = source.dbr().lower_warning_limit();
  dest.lower_alarm_limit = source.dbr().lower_alarm_limit();
  dest.upper_ctrl_limit = source.dbr().upper_ctrl_limit();
  dest.lower_ctrl_limit = source.dbr().lower_ctrl_limit();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvTimeString &source,
                    Unroll::EpicsPvTimeString &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  dest.stamp = source.dbr().stamp();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvTimeShort &source,
                    Unroll::EpicsPvTimeShort &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  dest.stamp = source.dbr().stamp();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvTimeFloat &source,
                    Unroll::EpicsPvTimeFloat &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  dest.stamp = source.dbr().stamp();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvTimeEnum &source,
                    Unroll::EpicsPvTimeEnum &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  dest.stamp = source.dbr().stamp();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvTimeChar &source,
                    Unroll::EpicsPvTimeChar &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  dest.stamp = source.dbr().stamp();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvTimeLong &source,
                    Unroll::EpicsPvTimeLong &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  dest.stamp = source.dbr().stamp();
}

void copyToUnrolledExceptForValue(const Psana::Epics::EpicsPvTimeDouble &source,
                    Unroll::EpicsPvTimeDouble &dest) 
{
  dest.iPvId = source.pvId();
  dest.iDbrType = source.dbrType();
  dest.iNumElements = source.numElements();
  dest.status = source.dbr().status();
  dest.severity = source.dbr().severity();
  dest.stamp = source.dbr().stamp();
}

 

 
} // Translator
