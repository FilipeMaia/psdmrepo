#ifndef TRANSLATOR_H5OUTPUT_H
#define TRANSLATOR_H5OUTPUT_H

#include <set>
#include <map>
#include <string>
#include <list>
#include <typeinfo>

#include "boost/shared_ptr.hpp"

#include "MsgLogger/MsgLogger.h"
#include "psana/Module.h"
#include "hdf5pp/File.h"
#include "hdf5pp/Group.h"

#include "LusiTime/Time.h"

#include "PSEvt/TypeInfoUtils.h"

#include "Translator/TypeAliases.h"
#include "Translator/DataSetCreationProperties.h"
#include "Translator/HdfWriterMap.h"
#include "Translator/HdfWriterEventId.h"
#include "Translator/HdfWriterDamage.h"
#include "Translator/HdfWriterString.h"
#include "Translator/TypeSrcKeyH5GroupDirectory.h"
#include "Translator/EpicsH5GroupDirectory.h"
#include "Translator/EventKeyTranslation.h"
#include "Translator/LessEventIdPtrs.h"
#include "Translator/ChunkManager.h"
#include "Translator/H5GroupNames.h"
#include "Translator/HdfWriterCalib.h"

namespace Translator {

class LessEventKey {
public:
  bool operator()(const PSEvt::EventKey & a, const PSEvt::EventKey & b ) const { 
    return a < b; 
  }
};

/**
 *  @ingroup Translator
 *
 *  @brief Main module for hdf5 translation.
 *
 *  Invoke this module (Translator.H5Output) when running psana to translate
 *  Psana events into a hdf5 file.  Note, this module should be invoked last
 *  so that it can pick up any other user module event data that can be 
 *  translated, or that effects translation.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class H5Output : public Module {
public:
  enum SplitMode { NoSplit, Family, SplitScan };

  H5Output(std::string);
  virtual ~H5Output();

  virtual void beginJob(Event& evt, Env& env);
  virtual void beginRun(Event& evt, Env& env);
  virtual void beginCalibCycle(Event& evt, Env& env);
  virtual void event(Event& evt, Env& env);
  virtual void endCalibCycle(Event& evt, Env& env);
  virtual void endRun(Event& evt, Env& env);
  virtual void endJob(Event& evt, Env& env);

  friend class Translator::ChunkManager;

protected:  
  void readConfigParameters();
  template <typename T>
    T configReportIfNotDefault(const std::string &param, const T &defaultValue) const
    {
      T configValue = config(param,defaultValue);
      if ((configValue != defaultValue) and (not m_quiet)) {
        MsgLog(name(),info," param " << param << " = " << configValue << " (non-default value)");
      }
      return configValue;
    }
  std::list<std::string> configListReportIfNotDefault(const std::string &param, 
                                                      const std::list<std::string> &defaultValue) const;
  void addCalibStoreHdfWriters(HdfWriterMap &hdfWriters);
  void removeCalibStoreHdfWriters(HdfWriterMap &hdfWriters);
  void openH5OutputFile();
  void createNextConfigureGroup();
  void setEventVariables(Event &evt, Env &env);
  void addConfigTypes(TypeSrcKeyH5GroupDirectory &configGroupDirectory,
                      hdf5pp::Group & parentGroup);
  void createNextRunGroup();
  void createNextCalibCycleGroup();
  void lookForAndStoreCalibData();
  void eventImpl();
  void setDamageMapFromEvent();
  Pds::Damage getDamageForEventKey(const EventKey &eventKey);

  boost::shared_ptr<HdfWriterFromEvent> checkTranslationFilters(const EventKey &eventKey, 
                                                           bool checkForCalibratedKey);
  std::list<PSEvt::EventKey> getUpdatedConfigKeys();
  void setEventKeysToTranslate(std::list<EventKeyTranslation> & toTranslate,
                               std::list<PSEvt::EventKey> &filtered);
  
  void addToFilteredEventGroup(const std::list<EventKey> &eventKeys, const PSEvt::EventId &eventId);
  void closeH5File();

  bool srcIsFiltered(const Pds::Src &);
  bool keyIsFiltered(const std::string &key);

  void filterHdfWriterMap();
  void initializeSrcAndKeyFilters();
  std::string eventPosition();

  /// returns true if C++ type is an ndarray that the system can translate
  bool isNDArray(const type_info *typeInfoPtr);

  void checkForNewWriters();

private:
  hdf5pp::File m_h5file;
  ChunkManager m_chunkManager;
  size_t m_currentConfigureCounter;
  size_t m_currentRunCounter;
  size_t m_currentCalibCycleCounter;
  size_t m_currentEventCounter; // reset when a CalibCycle begins
  size_t m_filteredEventsThisCalibCycle;
  size_t m_maxSavedPreviousSplitEvents;
  hdf5pp::Group m_currentConfigureGroup;
  hdf5pp::Group m_currentRunGroup;
  hdf5pp::Group m_currentCalibCycleGroup;
  hdf5pp::Group m_currentFilteredGroup;

  TypeSrcKeyH5GroupDirectory m_configureGroupDir;
  TypeSrcKeyH5GroupDirectory m_calibCycleConfigureGroupDir;
  TypeSrcKeyH5GroupDirectory m_calibCycleEventGroupDir;
  TypeSrcKeyH5GroupDirectory m_calibCycleFilteredGroupDir;
  TypeSrcKeyH5GroupDirectory m_calibStoreGroupDir;
  EpicsH5GroupDirectory m_epicsGroupDir;

  boost::shared_ptr<HdfWriterEventId> m_hdfWriterEventId;
  boost::shared_ptr<HdfWriterDamage> m_hdfWriterDamage;

  boost::shared_ptr<PSEvt::DamageMap> m_damageMap;
  Event *m_event;
  Env *m_env;
  boost::shared_ptr<EventId> m_eventId;

  std::map<PSEvt::EventKey, long>  m_configStoreUpdates;
  long m_totalConfigStoreUpdates;

  struct BlankNonBlanks {
    std::map<PSEvt::EventKey, long, LessEventKey> blanks;
    std::set<PSEvt::EventKey, LessEventKey> nonblanks;
  };
  typedef std::map< boost::shared_ptr<PSEvt::EventId>,  
                    BlankNonBlanks, 
                    LessEventIdPtrs > BlankNonBlanksMap;

  BlankNonBlanksMap m_previousSplitEvents;

  HdfWriterMap m_hdfWriters;

  TypeAliases m_typeAliases;

  bool m_includeAllSrc;
  bool m_includeAllKey;

  bool m_srcFilterIsExclude;
  bool m_keyFilterIsExclude;

  std::vector<PSEvt::Source::SrcMatch> m_psevtSourceFilterList;
  std::set<std::string> m_keyFilterSet;

  EpicsH5GroupDirectory::EpicsStoreMode m_storeEpics;

  /////////////////////////////////
  // parameters read in from config:
  // key parameters 
  std::string m_h5fileName;
  SplitMode m_split;
  hsize_t m_splitSize;

  bool m_short_bld_name;
  bool m_create_alias_links;
  bool m_overwrite;

  // translation parameters
  std::map<std::string, bool> m_typeInclude;  // each type alias will be read in 
                                              // and true if we convert that type
  std::list<std::string> m_src_filter;
  std::list<std::string> m_type_filter;
  std::list<std::string> m_key_filter; 

  std::vector<PSEvt::Source> m_SourceFilterList;

  std::string m_calibration_key;
  bool m_include_uncalibrated_data;
  bool m_exclude_calibrated_data;
  bool m_exclude_calibstore;

  std::set<PSEvt::EventKey, LessEventKey> m_calibratedEventKeys;

  bool m_quiet;

  bool m_defaultShuffle, m_eventIdShuffle, m_damageShuffle, 
    m_stringShuffle, m_epicsPvShuffle;

  int m_defaultDeflate, m_eventIdDeflate, m_damageDeflate, 
    m_stringDeflate, m_epicsPvDeflate;

  DataSetCreationProperties m_eventIdCreateDsetProp;
  DataSetCreationProperties m_damageCreateDsetProp;
  DataSetCreationProperties m_stringCreateDsetProp;
  DataSetCreationProperties m_epicsPvCreateDsetProp;
  DataSetCreationProperties m_defaultCreateDsetProp;
  DataSetCreationProperties m_ndarrayCreateDsetProp;

  boost::shared_ptr<H5GroupNames> m_h5groupNames;

  LusiTime::Time m_startTime, m_endTime;
}; // class H5Output

} // namespace

#endif  // TRANSLATOR_H5OUTPUT_H
