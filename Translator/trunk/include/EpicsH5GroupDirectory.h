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

class EpicsH5GroupDirectory {
 public:
  EpicsH5GroupDirectory();
  void initialize(boost::shared_ptr<HdfWriterEventId> hdfWriterEventId,
                  const DataSetCreationProperties & epicsPvHCreateDsetProp);
  void processBeginJob(hid_t currentConfigGroup, 
                       PSEnv::EpicsStore &epicsStore,
                       boost::shared_ptr<PSEvt::EventId> eventId);
  void processBeginCalibCycle(hid_t currentCalibCycleGroup, PSEnv::EpicsStore &epicsStore);
  void processEvent(PSEnv::EpicsStore & epicsStore, 
                    boost::shared_ptr<PSEvt::EventId> eventId);
  void processEndCalibCycle();
  void processEndJob();

 private:
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

  // For epics that occur outside of the configure 
  // transition, we map epics pvid 2 timestamp of the last time
  // the epics was written.
  std::map<std::string, Unroll::epicsTimeStamp> m_lastWriteMap;

  typedef enum {unknown, hasEpics, noEpics} EpicsStatus;
  EpicsStatus m_epicsStatus;

  bool m_epicsTypeAndSrcGroupsCreatedForThisCalibCycle;
  std::map<std::string, std::vector<std::string> > m_epicsPv2Aliases;
};

}

#endif
