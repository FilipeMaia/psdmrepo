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
#include "Translator/EpicsWriteBuffer.h"

namespace Translator {

/**
 *  @ingroup Translator
 *
 *  @brief class to write epics pv datasets, and the eventId datasets into hdf5 groups.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class HdfWriterEpicsPv {
 public:
  HdfWriterEpicsPv(const DataSetCreationProperties & oneElemDataSetCreationProperties, 
                   const DataSetCreationProperties & manyElemDataSetCreationProperties, 
                   boost::shared_ptr<HdfWriterEventId> );
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

  void closeDataset(hid_t groupId);

  const DataSetCreationProperties & oneElemDataSetCreationProperties() 
  { return m_oneElemDataSetCreationProperties; }

  const DataSetCreationProperties & manyElemDataSetCreationProperties() 
  { return m_manyElemDataSetCreationProperties; }

  void setOneElemDatasetCreationProperties(const DataSetCreationProperties & dataSetCreationProperties) 
  { m_oneElemDataSetCreationProperties = dataSetCreationProperties; }

  void setManyElemDatasetCreationProperties(const DataSetCreationProperties & dataSetCreationProperties) 
  { m_manyElemDataSetCreationProperties = dataSetCreationProperties; }

  typedef enum {CreateWriteClose, CreateAppend, Append} DispatchAction;

 protected:
  void dispatch(hid_t groupId, int16_t dbrType, 
                PSEnv::EpicsStore & epicsStore, 
                const std::string & pvName, 
                boost::shared_ptr<PSEvt::EventId> eventId,
                DispatchAction dispatchAction);
  
 private:
  const static int MANY_ELEM = 50;
  DataSetCreationProperties m_oneElemDataSetCreationProperties;
  DataSetCreationProperties m_manyElemDataSetCreationProperties;

  boost::shared_ptr<HdfWriterGeneric> m_hdfWriterGeneric;
  boost::shared_ptr<HdfWriterEventId> m_hdfWriterEventId;

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
    
    EpicsWriteBuffer<U> epicsWriteBuffer(dbrType, *psanaVar);

    try {
      hid_t fileTypeId = epicsWriteBuffer.getFileH5Type();
      hid_t memTypeId = epicsWriteBuffer.getMemH5Type();
      size_t dsetIdx = -1;
      if (not eventId) MsgLog("Translator.HdfWriterEpicsPv.doDispatch",fatal,"null eventId: dispatchAction=" << dispatchAction);
      switch (dispatchAction) {
      case CreateWriteClose:
        dsetIdx = m_hdfWriterGeneric->createFixedSizeDataset(groupId, "data", 
                                                             fileTypeId, memTypeId,1);
        m_hdfWriterGeneric->append(groupId, dsetIdx, epicsWriteBuffer.data());
        m_hdfWriterGeneric->closeDatasets(groupId);
        m_hdfWriterEventId->make_dataset(groupId);
        m_hdfWriterEventId->append(groupId, *eventId);
        m_hdfWriterEventId->closeDataset(groupId);
      
        break;

      case CreateAppend:
        if (psanaVar->numElements() > MANY_ELEM) {
          dsetIdx = m_hdfWriterGeneric->createUnlimitedSizeDataset(groupId, "data", 
                                                                   fileTypeId, memTypeId,
                                                                   manyElemDataSetCreationProperties());
        } else {
          dsetIdx = m_hdfWriterGeneric->createUnlimitedSizeDataset(groupId, "data", 
                                                                   fileTypeId, memTypeId,
                                                                   oneElemDataSetCreationProperties());
        }
        m_hdfWriterGeneric->append(groupId, dsetIdx, epicsWriteBuffer.data());
        m_hdfWriterEventId->make_dataset(groupId);
        m_hdfWriterEventId->append(groupId, *eventId);
        
        break;
        
      case Append:
        dsetIdx = 0; // we only use one dataset for epics pv's
        m_hdfWriterGeneric->append(groupId, dsetIdx, epicsWriteBuffer.data());
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
  public: Exception(const ErrSvc::Context &ctx, const std::string &what) 
     : ErrSvc::Issue(ctx,what) {}
  };
};

 std::ostream & operator<<(std::ostream &, HdfWriterEpicsPv::DispatchAction da);
} // namespace

#endif 
