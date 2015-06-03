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
#include "Translator/SplitScanMgr.h"

namespace Translator {

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
  H5Output(std::string);
  virtual ~H5Output();
 
  virtual void beginJob(Event& evt, Env& env);
  virtual void beginRun(Event& evt, Env& env);
  virtual void beginCalibCycle(Event& evt, Env& env);
  virtual void event(Event& evt, Env& env);
  virtual void endCalibCycle(Event& evt, Env& env);
  virtual void endRun(Event& evt, Env& env);
  virtual void endJob(Event& evt, Env& env);

  boost::shared_ptr<SplitScanMgr> splitScanMgr() { return m_splitScanMgr; }

  hdf5pp::Group currentRunGroup() { return m_currentRunGroup; }

  hdf5pp::Group currentConfigureGroup() { return m_currentConfigureGroup; }

  // returns call to H5Fflush if file is valid, -1 if it is not. Calls MsgLog with error if problem
  herr_t flushOutputFile();

  /// name for the CalibStore in the hdf5 file
  static const std::string calibStoreGroupName; 

  friend class Translator::ChunkManager;

protected:  

  void init();
  void createH5OutputFile();

  /// returns output_file from config - may need to be changed if in split scan mode
  std::string readConfigParameters();

  static const std::string & msgLoggerName();

  template <typename T>
    T configReportIfNotDefault(const std::string &param, const T &defaultValue) const
    {
      T configValue = config(param,defaultValue);
      if ((configValue != defaultValue) and (not m_quiet)) {
        MsgLog(msgLoggerName(),info," param " << param << " = " << configValue << " (non-default value)");
      }
      return configValue;
    }
  std::list<std::string> configListReportIfNotDefault(const std::string &param, 
                                                      const std::list<std::string> &defaultValue) const;
  void addCalibStoreHdfWriters(HdfWriterMap &hdfWriters);
  void removeCalibStoreHdfWriters(HdfWriterMap &hdfWriters);
  void createNextConfigureGroup(boost::shared_ptr<EventId> eventId);
  void addConfigTypes(PSEvt::Event &evt,
		      PSEnv::Env &env,
		      TypeSrcKeyH5GroupDirectory &configGroupDirectory,
                      hdf5pp::Group & parentGroup);
  void createNextRunGroup(boost::shared_ptr<EventId> eventId);
  void createNextCalibCycleGroup(boost::shared_ptr<EventId> eventId);
  void createNextCalibCycleExtLink(const char *linkName, hdf5pp::Group &runGroup);
  void lookForAndStoreCalibData(PSEvt::Event &evt, PSEnv::Env &env, hdf5pp::Group &parentGroup);
  void lookForAndStoreEndData(PSEvt::Event &evt, PSEnv::Env &env, hdf5pp::Group &parentGroup, 
                              TypeSrcKeyH5GroupDirectory & endDataGroupDir, hdf5pp::Group &endDataGroup);
  void eventImpl(PSEvt::Event &evt, PSEnv::Env &env);
  Pds::Damage getDamageForEventKey(const EventKey &eventKey, 
				   boost::shared_ptr<PSEvt::DamageMap> damageMap);

  // last argument is output argument, typeclass of found filter, or UnknownType if not found
  boost::shared_ptr<HdfWriterFromEvent> checkTranslationFilters(PSEvt::Event &evt,
                                                                const EventKey &eventKey, 
                                                                bool checkForCalibratedKey,
                                                                TypeClass &typeClass);
  std::list<PSEvt::EventKey> getUpdatedConfigKeys(PSEnv::Env &env);
  void setEventKeysToTranslate(PSEvt::Event &evt, PSEnv::Env &env,
                               std::list<EventKeyTranslation> & toTranslate,
                               bool & eventIsFiltered);
  
  // this is for closing the file during an Event exception
  void closeH5FileDueToEventException();

  bool srcIsFiltered(const Pds::Src &src);
  bool keyIsFiltered(const std::string &key);
  bool fullEventKeyIsFiltered(const std::type_info *typeInfoPtr, const Pds::Src &src, const std::string &key);

  // returns true is type filter is exclude psana - caller should set epics to no if so
  bool filterHdfWriterMap();
  void initializeSrcAndKeyFilters(PSEnv::Env &env);
  void initializeEventKeyFilter(PSEnv::Env &env);
  std::string eventPosition();

  /// returns true if C++ type is an ndarray that the system can translate
  bool isNDArray(const std::type_info *typeInfoPtr);

  void checkForNewWriters(PSEvt::Event &evt);
  bool checkIfNewTypeHasSameH5GroupNameAsCurrentTypes(const std::type_info *);

  void reportRunTime();

  const DataSetCreationProperties & lookUpDataSetCreationProp(TypeClass typeClass);

private:
  hdf5pp::File m_h5file;
  ChunkManager m_chunkManager;
  size_t m_currentConfigureCounter;
  size_t m_currentRunCounter;
  size_t m_currentCalibCycleCounter;
  size_t m_currentEventCounter; // reset when a CalibCycle begins
  size_t m_totalEventsProcessed;
  size_t m_totalCalibCyclesProcessed;
  size_t m_minEventsPerMPIWorker;
  size_t m_maxSavedPreviousSplitEvents;
  hdf5pp::Group m_currentConfigureGroup;
  hdf5pp::Group m_currentRunGroup;
  hdf5pp::Group m_currentCalibCycleGroup;
  hdf5pp::Group m_currentCalibCycleEndGroup;
  hdf5pp::Group m_currentRunEndGroup;
  hdf5pp::Group m_currentConfigureEndGroup;

  TypeSrcKeyH5GroupDirectory m_configureGroupDir;
  TypeSrcKeyH5GroupDirectory m_runGroupDir;
  TypeSrcKeyH5GroupDirectory m_calibCycleGroupDir;
  TypeSrcKeyH5GroupDirectory m_calibCycleEndGroupDir;
  TypeSrcKeyH5GroupDirectory m_runEndGroupDir;
  TypeSrcKeyH5GroupDirectory m_configureEndGroupDir;
  TypeSrcKeyH5GroupDirectory m_calibStoreGroupDir;
  EpicsH5GroupDirectory m_epicsGroupDir;

  boost::shared_ptr<HdfWriterEventId> m_hdfWriterEventId;
  boost::shared_ptr<HdfWriterDamage> m_hdfWriterDamage;

  boost::shared_ptr<EventId> m_eventId;

  std::map<PSEvt::EventKey, long>  m_configStoreUpdates;
  long m_totalConfigStoreUpdates;

  struct BlankNonBlanks {
    std::map<PSEvt::EventKey, long> blanks;
    std::set<PSEvt::EventKey> nonblanks;
  };
  typedef std::map< boost::shared_ptr<PSEvt::EventId>,  
                    BlankNonBlanks, 
                    LessEventIdPtrs > BlankNonBlanksMap;

  BlankNonBlanksMap m_previousSplitEvents;

  HdfWriterMap m_hdfWriters;

  TypeAliases m_typeAliases;

  bool m_includeAllSrc;
  bool m_includeAllKey;
  bool m_includeAllEventKey;

  bool m_srcFilterIsExclude;
  bool m_keyFilterIsExclude;
  bool m_eventkeyFilterIsExclude;

  std::vector<PSEvt::Source::SrcMatch> m_psevtSourceFilterList;
  std::set<std::string> m_keyFilterSet;
  typedef std::vector< std::pair<PSEvt::Source::SrcMatch, std::string> > SrcKeyList; 
  std::map<const std::type_info *, SrcKeyList>  m_eventKeyFilters;

  EpicsH5GroupDirectory::EpicsStoreMode m_storeEpics;

  /////////////////////////////////
  // parameters read in from config:
  std::string m_h5fileName;  // this may be changed from the config parameter if this is a MPI worker
  bool m_overwrite;

  SplitScanMgr::SplitMode m_split;
  bool m_splitCCInSubDir;
  int m_mpiWorkerStartCalibCycle;
  hsize_t m_splitSize;  

  // translation parameters
  std::map<std::string, bool> m_typeInclude;  // each type alias will be read in 
                                              // and true if we convert that type
  std::list<std::string> m_src_filter;
  bool m_unknown_src_ok;
  std::list<std::string> m_type_filter;
  std::list<std::string> m_key_filter; 
  std::list<std::string> m_eventkey_filter; 

  std::vector<PSEvt::Source> m_SourceFilterList;

  std::string m_calibration_key;
  bool m_skip_calibrated;
  bool m_exclude_calibstore;

  std::set<PSEvt::EventKey> m_calibratedEventKeys;

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
  double m_translatorTime;
  boost::shared_ptr<SplitScanMgr> m_splitScanMgr;
  bool m_printedNotFilteringWarning;
}; // class H5Output

} // namespace

#endif  // TRANSLATOR_H5OUTPUT_H
