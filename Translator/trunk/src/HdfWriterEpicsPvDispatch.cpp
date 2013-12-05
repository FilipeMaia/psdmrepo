
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/epics.ddl.h"
#include "Translator/epics.ddl.h"
#include "Translator/HdfWriterEpicsPv.h"

using namespace Translator;

namespace {
  const char * logger = "HdfWriterEpicsPv";
}

void HdfWriterEpicsPv::dispatch(hid_t groupId, int16_t dbrType, 
                                PSEnv::EpicsStore & epicsStore, 
                                const std::string & epicsPvName,
                                boost::shared_ptr<PSEvt::EventId> eventId,
                                DispatchAction dispatchAction) {
  switch (dbrType) {
  case Psana::Epics::DBR_CTRL_STRING:
    doDispatchAction<Unroll::EpicsPvCtrlString>(dbrType, "DBR_CTRL_STRING", 
                             "Psana::Epics::EpicsPvCtrlString",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_CTRL_SHORT:
    doDispatchAction<Unroll::EpicsPvCtrlShort>(dbrType, "DBR_CTRL_SHORT", 
                             "Psana::Epics::EpicsPvCtrlShort",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_CTRL_FLOAT:
    doDispatchAction<Unroll::EpicsPvCtrlFloat>(dbrType, "DBR_CTRL_FLOAT", 
                             "Psana::Epics::EpicsPvCtrlFloat",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_CTRL_ENUM:
    doDispatchAction<Unroll::EpicsPvCtrlEnum>(dbrType, "DBR_CTRL_ENUM", 
                             "Psana::Epics::EpicsPvCtrlEnum",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_CTRL_CHAR:
    doDispatchAction<Unroll::EpicsPvCtrlChar>(dbrType, "DBR_CTRL_CHAR", 
                             "Psana::Epics::EpicsPvCtrlChar",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_CTRL_LONG:
    doDispatchAction<Unroll::EpicsPvCtrlLong>(dbrType, "DBR_CTRL_LONG", 
                             "Psana::Epics::EpicsPvCtrlLong",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_CTRL_DOUBLE:
    doDispatchAction<Unroll::EpicsPvCtrlDouble>(dbrType, "DBR_CTRL_DOUBLE", 
                             "Psana::Epics::EpicsPvCtrlDouble",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_TIME_STRING:
    doDispatchAction<Unroll::EpicsPvTimeString>(dbrType, "DBR_TIME_STRING", 
                             "Psana::Epics::EpicsPvTimeString",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_TIME_SHORT:
    doDispatchAction<Unroll::EpicsPvTimeShort>(dbrType, "DBR_TIME_SHORT", 
                             "Psana::Epics::EpicsPvTimeShort",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_TIME_FLOAT:
    doDispatchAction<Unroll::EpicsPvTimeFloat>(dbrType, "DBR_TIME_FLOAT", 
                             "Psana::Epics::EpicsPvTimeFloat",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_TIME_ENUM:
    doDispatchAction<Unroll::EpicsPvTimeEnum>(dbrType, "DBR_TIME_ENUM", 
                             "Psana::Epics::EpicsPvTimeEnum",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_TIME_CHAR:
    doDispatchAction<Unroll::EpicsPvTimeChar>(dbrType, "DBR_TIME_CHAR", 
                             "Psana::Epics::EpicsPvTimeChar",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_TIME_LONG:
    doDispatchAction<Unroll::EpicsPvTimeLong>(dbrType, "DBR_TIME_LONG", 
                             "Psana::Epics::EpicsPvTimeLong",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  case Psana::Epics::DBR_TIME_DOUBLE:
    doDispatchAction<Unroll::EpicsPvTimeDouble>(dbrType, "DBR_TIME_DOUBLE", 
                             "Psana::Epics::EpicsPvTimeDouble",
                             groupId, epicsStore, epicsPvName, 
                             eventId, dispatchAction);
    break;
  default:
    MsgLog(logger, warning, "unexpected dbr type: " << dbrType << " in dispatch");
  }
}
