#ifndef TRANSLATOR_HDFWRITEREPICSPV_H
#define TRANSLATOR_HDFWRITEREPICSPV_H

#include <map>
#include <sstream>
#include <iostream>

#include "hdf5/hdf5.h"
#include "boost/shared_ptr.hpp"

#include "PSEvt/EventId.h"
#include "PSEnv/EpicsStore.h"
#include "psddl_psana/epics.ddl.h"
#include "MsgLogger/MsgLogger.h"
#include "Translator/epics.ddl.h"
#include "Translator/DataSetCreationProperties.h"
#include "Translator/HdfWriterGeneric.h"
#include "Translator/HdfWriterEventId.h"

namespace Translator {

class HdfWriterEpicsPv {
 public:
  HdfWriterEpicsPv(const DataSetCreationProperties &, boost::shared_ptr<HdfWriterEventId> );
  ~HdfWriterEpicsPv();

  void oneTimeCreateAndWrite(hid_t groupId, int16_t dbrType, 
                             PSEnv::EpicsStore & epicsStore, 
                             const std::string & pvName, 
                             boost::shared_ptr<PSEvt::EventId> eventId) { 
    dispatch(groupId, dbrType, epicsStore, pvName, eventId, CreateWriteClose);
  }
  void createAndAppend(hid_t groupId, int16_t dbrType, 
                       PSEnv::EpicsStore & epicsStore, 
                       const std::string & pvName,
                       boost::shared_ptr<PSEvt::EventId> eventId) {
    dispatch(groupId, dbrType, epicsStore, pvName, eventId, CreateAppend);
  }
  void append(hid_t groupId, int16_t dbrType, 
              PSEnv::EpicsStore & epicsStore, 
              const std::string & pvName,
              boost::shared_ptr<PSEvt::EventId> eventId) {
    dispatch(groupId, dbrType, epicsStore,pvName, eventId, Append);
  }
  void closeDataset(hid_t groupId) { m_hdfWriterGeneric->closeDatasets(groupId); }
  
  const DataSetCreationProperties & dataSetCreationProperties() 
  { return m_dataSetCreationProperties; }
  void setDatasetCreationProperties(const DataSetCreationProperties & dataSetCreationProperties) 
  { m_dataSetCreationProperties = dataSetCreationProperties; }

  typedef enum {CreateWriteClose, CreateAppend, Append} DispatchAction;

 protected:
  void makeSharedTypes();
  void closeSharedTypes();
  hid_t getTypeId(int16_t dbrType);
  void dispatch(hid_t groupId, int16_t dbrType, 
                PSEnv::EpicsStore & epicsStore, 
                const std::string & pvName, 
                boost::shared_ptr<PSEvt::EventId> eventId,
                DispatchAction dispatchAction);
  
 private:
  DataSetCreationProperties m_dataSetCreationProperties;

  boost::shared_ptr<HdfWriterGeneric> m_hdfWriterGeneric;
  boost::shared_ptr<HdfWriterEventId> m_hdfWriterEventId;
  std::map<uint16_t, hid_t>  m_dbr2h5TypeId;   
  
  // base h5 types that are used in the Epics Pv types
  hid_t m_pvNameType;  // char[Psana::Epics::iMaxPvNameLength
  hid_t m_stringType;  // char[Psana::Epics:: MAX_STRING_SIZE]
  hid_t m_unitsType;   // char[Psana::Epics::MAX_UNITS_SIZE]
  hid_t m_enumStrType; // char[Psana::Epics::MAX_ENUM_STRING_SIZE]
  hid_t m_allEnumStrsType; // array of [Psana::Epics::MAX_ENUM_STATES] m_enumStrType
  hid_t m_stampType;

  template <class U>
  void doDispatchAction(int16_t dbrType,
                        const std::string &dbrStr, 
                        const std::string &psanaTypeStr,
                        hid_t groupId, 
                        PSEnv::EpicsStore & epicsStore,
                        const std::string &epicsPvName, 
                        boost::shared_ptr<PSEvt::EventId> eventId,
                        DispatchAction dispatchAction) {
    typedef typename U::PsanaSrc PsanaSrc;
    boost::shared_ptr<PsanaSrc> psanaVar = epicsStore.getPV(epicsPvName);
    if (not psanaVar) MsgLog("Translator.HdfWriterEpicsPv.doDispatch",fatal,
                             "dbr type is " << dbrStr 
                             << " but cannot get psana type " << psanaTypeStr
                             << " from epicsStore. groupId=" << groupId
                             << " dispatchAction=" << dispatchAction);
    MsgLog("Translator.HdfWriterEpicsPv.doDispatch",debug,
           "dbrType=" << dbrType << " dbrStr=" << dbrStr 
           << " psanaTypeStr=" << psanaTypeStr << " groupId=" << groupId
           << " epicsPvName=" << epicsPvName << " dispatchAction=" << dispatchAction);
    
    U unrollBuffer;
    
    try {
      hid_t typeId = -1;
      size_t dsetIdx = -1;
      int16_t el = -1;
      switch (dispatchAction) {
      case CreateWriteClose:
        if (psanaVar->numElements()>1) MsgLog("Translator.HdfWriterEpicsPv",trace,"pv with " << 
                                              psanaVar->numElements() << " elements");
        typeId = getTypeId(dbrType);
        dsetIdx = m_hdfWriterGeneric->createFixedSizeDataset(groupId, "data", 
                                                             typeId,
                                                             psanaVar->numElements());
        copyToUnrolled(*psanaVar, 0, unrollBuffer);
        for (el = 0; el < psanaVar->numElements(); ++el) {
          if (el>0) copyValueFldToUnrolled<U>(*psanaVar, el, unrollBuffer);
          m_hdfWriterGeneric->append(groupId, dsetIdx, &unrollBuffer);
        }
        m_hdfWriterGeneric->closeDatasets(groupId);
        m_hdfWriterEventId->make_dataset(groupId);
        m_hdfWriterEventId->append(groupId, *eventId);
        m_hdfWriterEventId->closeDataset(groupId);
      
        break;

      case CreateAppend:
        typeId = getTypeId(dbrType);
        dsetIdx = m_hdfWriterGeneric->createUnlimitedSizeDataset(groupId, "data", 
                                                                 typeId,
                                                                 dataSetCreationProperties());
        copyToUnrolled(*psanaVar, 0, unrollBuffer);
        for (el = 0; el < psanaVar->numElements(); ++el) {
          if (el>0) copyValueFldToUnrolled<U>(*psanaVar, el, unrollBuffer);
          m_hdfWriterGeneric->append(groupId, dsetIdx, &unrollBuffer);
        }
        
        m_hdfWriterEventId->make_dataset(groupId);
        m_hdfWriterEventId->append(groupId, *eventId);
        
        break;
        
      case Append:
        dsetIdx = 0; // we only use one dataset for epics pv's
        copyToUnrolled(*psanaVar, 0, unrollBuffer);
        for (int16_t el = 0; el < psanaVar->numElements(); ++el) {
          if (el>0) copyValueFldToUnrolled<U>(*psanaVar, el, unrollBuffer);
          m_hdfWriterGeneric->append(groupId, dsetIdx, &unrollBuffer);
        }
        
        m_hdfWriterEventId->append(groupId, *eventId);
        
        break;
      }
    } catch (ErrSvc::Issue &issue) {
      std::cout << issue.what();
      std::ostringstream msg;
      msg <<  "dbrType=" << dbrType << " dbrStr=" << dbrStr 
          << " psanaTypeStr=" << psanaTypeStr << " groupId=" << groupId
          << " epicsPvName=" << epicsPvName << " dispatchAction=" << dispatchAction;
      throw Exception(ERR_LOC, msg.str());
    }
  };
 public:
 class Exception : public ErrSvc::Issue {
  public: Exception(const ErrSvc::Context &ctx, const std::string &what) : ErrSvc::Issue(ctx,what) {}
  };
};

 std::ostream & operator<<(std::ostream &, HdfWriterEpicsPv::DispatchAction da);
} // namespace

#endif 
