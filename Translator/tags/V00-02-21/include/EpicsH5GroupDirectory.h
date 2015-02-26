#ifndef TRANSLATOR_EPICSH5GROUPDIRECTORY_H
#define TRANSLATOR_EPICSH5GROUPDIRECTORY_H

#include <string>
#include <map>
#include <vector>

#include "hdf5/hdf5.h"
#include "PSEnv/EpicsStore.h"
#include "PSEvt/DamageMap.h"
#include "PSEvt/EventId.h"
#include "Translator/DataSetCreationProperties.h"
#include "Translator/HdfWriterEventId.h"
#include "Translator/HdfWriterEpicsPv.h"
#include "Translator/epics.ddl.h"

namespace Translator {

/**
 *  @ingroup Translator
 *
 *  @brief Manages the epics groups in both the Configure and CalibCycle's
 *
 *  An instance of this class should be created to translate epics pv's in the
 *  Psana configStore.  It creates and uses an instance of HdfWriterEpicsPv for 
 *  the details of managing and writing to the epics datasets, while this class 
 *  manages the epics pv groups that are created for all the pv's.  
 *
 *  The processBeginJob, processBeginCalibCycle functions are passed the 
 *  hdf5 groups that will be the parent groups to the epics groups.
 *  processEvent is called with each event.  This class remembers the epics
 *  time stamp of the last pv seen.  With each call to processEvent, it goes through
 *  all the epics pv's in the config store and checks for a new timestamp.
 * 
 *  @note This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */  
class EpicsH5GroupDirectory {
 public:
  typedef enum {Unknown, DoNotStoreEpics, RepeatEpicsEachCalib, 
                OnlyStoreEpicsUpdates, StoreAllEpicsOnEveryShot} EpicsStoreMode;
  static std::string epicsStoreMode2str(const EpicsStoreMode storeMode);

  EpicsH5GroupDirectory();
  void initialize(EpicsStoreMode epicsStoreMode,
                  boost::shared_ptr<HdfWriterEventId> hdfWriterEventId,
                  const DataSetCreationProperties & oneElemEpicsPvHCreateDsetProp,
                  const DataSetCreationProperties & manyElemEpicsPvHCreateDsetProp);
  void processBeginJob(hid_t currentConfigGroup, 
                       PSEnv::EpicsStore &epicsStore,
                       boost::shared_ptr<PSEvt::EventId> eventId);
  void processBeginCalibCycle(hid_t currentCalibCycleGroup, PSEnv::EpicsStore &epicsStore);
  void processEvent(PSEnv::EpicsStore & epicsStore, 
                    boost::shared_ptr<PSEvt::EventId> eventId);
  void processEndCalibCycle();
  void processEndJob();

  bool writeCurrentPv(long currentEventTag, long previousEventTag);

 private:
  EpicsStoreMode m_epicsStoreMode;
  bool checkIfStoringEpics();
  boost::shared_ptr<HdfWriterEpicsPv> m_hdfWriterEpicsPv;

  hid_t m_configureGroup;
  hid_t m_currentCalibCycleGroup;

  hid_t m_configEpicsTypeGroup;
  hid_t m_configEpicsSrcGroup;
  std::map<std::string, hid_t> m_configEpicsPvGroups;

  hid_t m_calibCycleEpicsTypeGroup;
  hid_t m_calibCycleEpicsSrcGroup;
  // map epics pv name to group id
  std::map<std::string, hid_t> m_calibEpicsPvGroups;

  // keep For epics that occur outside of the configure 
  // transition, we map epics pvid 2 timestamp of the last time
  // the epics was written.
  std::map<std::string, long> m_lastPvEventTag;

  typedef enum {unknown, hasEpics, noEpics} EpicsStatus;
  EpicsStatus m_epicsStatus;

  bool m_epicsTypeAndSrcGroupsCreatedForThisCalibCycle;
  std::map<std::string, std::vector<std::string> > m_epicsPv2Aliases;
};

}

#endif
