#ifndef TRANSLATOR_H5OUTPUT_H
#define TRANSLATOR_H5OUTPUT_H

#include <set>
#include <map>
#include <string>
#include <list>
#include <typeinfo>

#include "boost/shared_ptr.hpp"

#include "psana/Module.h"
#include "hdf5pp/File.h"
#include "hdf5pp/Group.h"
#include "psddl_hdf2psana/ChunkPolicy.h"

#include "PSEvt/TypeInfoUtils.h"

#include "Translator/TypeAliases.h"
#include "Translator/DataSetCreationProperties.h"
#include "Translator/HdfWriterMap.h"
#include "Translator/HdfWriterEventId.h"
#include "Translator/HdfWriterDamage.h"
#include "Translator/HdfWriterFilterMsg.h"
#include "Translator/TypeSrcKeyH5GroupDirectory.h"
#include "Translator/EpicsH5GroupDirectory.h"
#include "Translator/EventKeyTranslation.h"
#include "Translator/LessEventIdPtrs.h"
/**
   Defines the H5Output module.
   This is a Psana module that will write Psana events to a
   Hdf5 file.
 */

namespace Translator {

class LessEventKey {
public:
  bool operator()(const PSEvt::EventKey & a, const PSEvt::EventKey & b ) const { 
    return a < b; 
  }
};

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

protected:  
  void readConfigParameters();
  void initializeCalibratedTypes();
  void openH5OutputFile();
  void createNextConfigureGroup();
  void setEventVariables(Event &evt, Env &env);
  void addConfigTypes(TypeSrcKeyH5GroupDirectory &configGroupDirectory,
                      hdf5pp::Group & parentGroup);
  void createNextRunGroup();
  void createNextCalibCycleGroup();
  void eventImpl();
  void setDamageMapFromEvent();
  Pds::Damage getDamageForEventKey(const EventKey &eventKey);

  boost::shared_ptr<HdfWriterBase> checkTranslationFilters(const EventKey &eventKey, 
                                                           bool checkForCalibratedKey);
  std::list<PSEvt::EventKey> getUpdatedConfigKeys();
  std::list<EventKeyTranslation> setEventKeysToTranslate(bool checkForCalibratedKey);
  
  void addToFilteredEventDataset(const PSEvt::EventId &eventId, const std::string &msg);
  void closeH5File();
  bool checkForAndProcessExcludeEvent();
  bool doNotTranslate(const Pds::Src &);
  bool doTranslate(const Pds::Src &src) { return not doNotTranslate(src); }
  void filterHdfWriterMap();
  void initializeSrcFilter();
  std::string eventPosition();
private:
  hdf5pp::File m_h5file;
  static const int m_h5schema = 3;

  size_t m_currentConfigureCounter;
  size_t m_currentRunCounter;
  size_t m_currentCalibCycleCounter;
  size_t m_currentEventCounter; // reset when a CalibCycle begins
  size_t m_filteredEventsThisCalibCycle;
  size_t m_maxSavedPreviousSplitEvents;
  hdf5pp::Group m_currentConfigureGroup;
  hdf5pp::Group m_currentRunGroup;
  hdf5pp::Group m_currentCalibCycleGroup;
  hid_t m_currentFilteredGroup;

  TypeSrcKeyH5GroupDirectory m_configureGroupDir;
  TypeSrcKeyH5GroupDirectory m_calibCycleConfigureGroupDir;
  TypeSrcKeyH5GroupDirectory m_calibCycleEventGroupDir;
  EpicsH5GroupDirectory m_epicsGroupDir;

  boost::shared_ptr<HdfWriterEventId> m_hdfWriterEventId;
  boost::shared_ptr<HdfWriterDamage> m_hdfWriterDamage;
  boost::shared_ptr<HdfWriterFilterMsg> m_hdfWriterFilterMsg;

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
  std::set<const std::type_info *, PSEvt::TypeInfoUtils::lessTypeInfoPtr> m_calibratedTypes;
  bool m_includeAllSrc;
  bool m_srcFilterIsExclude;
  std::set<std::string> m_srcNameFilterSet;
  bool m_storeEpics;

  /////////////////////////////////
  // parameters read in from config:
  // key parameters 
  std::string m_h5fileName;
  SplitMode m_split;
  hsize_t m_splitSize;

  bool m_short_bld_name;

  // translation parameters
  std::map<std::string, bool> m_typeInclude;  // each type alias will be read in and true if we convert that type
  std::list<std::string> m_src_filter;
  std::list<std::string> m_ndarray_eventid_keys_to_translate;  // do I need this?

  std::string m_calibration_key;
  bool m_include_uncalibrated_data;

  // default chunk parameters
  hsize_t m_chunkSizeInBytes;
  int m_chunkSizeInElements;
  hsize_t m_maxChunkSizeInBytes;
  int m_minObjectsPerChunk;
  int m_maxObjectsPerChunk;
  hsize_t m_minChunkCacheSize;
  hsize_t m_maxChunkCacheSize;

  // chunk parameters for specific datasets
  hsize_t m_eventIdChunkSizeInBytes;
  int m_eventIdChunkSizeInElements;

  hsize_t m_damageChunkSizeInBytes;
  int m_damageChunkSizeInElements;

  hsize_t m_filterMsgChunkSizeInBytes;
  int m_filterMsgChunkSizeInElements;
  
  hsize_t m_epicsPvChunkSizeInBytes;
  int m_epicsPvChunkSizeInElements;

  bool m_defaultShuffle, m_eventIdShuffle, m_damageShuffle, 
    m_filterMsgShuffle, m_epicsPvShuffle;

  int m_defaultDeflate, m_eventIdDeflate, m_damageDeflate, 
    m_filterMsgDeflate, m_epicsPvDeflate;

  boost::shared_ptr<psddl_hdf2psana::ChunkPolicy> m_defaultChunkPolicy;
  boost::shared_ptr<psddl_hdf2psana::ChunkPolicy> m_eventIdChunkPolicy;
  boost::shared_ptr<psddl_hdf2psana::ChunkPolicy> m_damageChunkPolicy;
  boost::shared_ptr<psddl_hdf2psana::ChunkPolicy> m_filterMsgChunkPolicy;
  boost::shared_ptr<psddl_hdf2psana::ChunkPolicy> m_epicsPvChunkPolicy;

  DataSetCreationProperties m_eventIdCreateDsetProp;
  DataSetCreationProperties m_damageCreateDsetProp;
  DataSetCreationProperties m_filterMsgCreateDsetProp;
  DataSetCreationProperties m_epicsPvCreateDsetProp;
  DataSetCreationProperties m_defaultCreateDsetProp;
}; // class H5Output

} // namespace

#endif  // TRANSLATOR_H5OUTPUT_H
