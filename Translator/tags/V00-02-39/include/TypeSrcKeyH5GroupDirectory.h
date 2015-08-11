#ifndef TRANSLATOR_TYPESRCKEYH5GROUPDIRECTORY_H
#define TRANSLATOR_TYPESRCKEYH5GROUPDIRECTORY_H

#include <set>
#include <map>
#include <list>
#include <string>
#include <functional>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "pdsdata/xtc/Src.hh"

#include "PSEvt/EventKey.h"
#include "PSEvt/TypeInfoUtils.h"
#include "PSEvt/Event.h"
#include "PSEnv/Env.h"

#include "hdf5pp/Group.h"

#include "Translator/HdfWriterFromEvent.h"
#include "Translator/HdfWriterEventId.h"
#include "Translator/HdfWriterDamage.h"
#include "Translator/DataSetCreationProperties.h"
#include "Translator/TypeAliases.h"
#include "Translator/H5GroupNames.h"

namespace Translator {

/**
 *  @ingroup Translator
 *
 *  @brief class to manage the datasets in a src group.
 *
 *  Instances of SrcKeyGroup are created and managed by TypeSrcKeyH5GroupDirectory,
 *  refer to that class for the top level interface for translated data to the hdf5
 *  file.
 * 
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class SrcKeyGroup {
 public:
 SrcKeyGroup() : 
    m_writtenForThisEvent(false),
    m_initialBlanks(0),
    m_totalEntries(0),
    m_datasetsCreated(None) {};

  SrcKeyGroup(hdf5pp::Group group, 
              const PSEvt::EventKey &eventKey,
              boost::shared_ptr<Translator::HdfWriterFromEvent> hdfWriter, 
              boost::shared_ptr<Translator::HdfWriterEventId> hdfWriterEventId,
              boost::shared_ptr<Translator::HdfWriterDamage> hdfWriterDamage) :     
    m_eventKey(eventKey),
    m_group(group), 
    m_hdfWriter(hdfWriter),
    m_hdfWriterEventId(hdfWriterEventId),
    m_hdfWriterDamage(hdfWriterDamage),
    m_writtenForThisEvent(false),
    m_initialBlanks(0),
    m_totalEntries(0),
    m_datasetsCreated(None) {};

  hdf5pp::Group & group() { return m_group; };
  void written(bool val) { m_writtenForThisEvent = val; };
  bool written() { return m_writtenForThisEvent; };
  void make_timeDamageDatasets();
  void make_typeDatasets(DataTypeLoc dataTypeLoc, PSEvt::Event &evt, PSEnv::Env &env, 
                     const Translator::DataSetCreationProperties &);
  void make_datasets(DataTypeLoc dataTypeLoc, PSEvt::Event &evt, PSEnv::Env &env, 
                     const Translator::DataSetCreationProperties &);
  void storeData(const PSEvt::EventKey & eventKey, DataTypeLoc dataTypeLoc, 
                 PSEvt::Event &evt, PSEnv::Env &env);
  long appendDataTimeAndDamage(const PSEvt::EventKey & eventKey, 
                               DataTypeLoc dataTypeLoc,
                               PSEvt::Event &evt, 
                               PSEnv::Env &env, 
                               boost::shared_ptr<PSEvt::EventId> eventId,
                               Pds::Damage damage);
  long appendBlankTimeAndDamage(const PSEvt::EventKey & eventKey, 
                                boost::shared_ptr<PSEvt::EventId> eventId,
                                Pds::Damage damage);
  void overwriteDataAndDamage(long index, 
                              const PSEvt::EventKey &eventKey, 
                              DataTypeLoc dataTypeLoc,
                              PSEvt::Event & evt, 
                              PSEnv::Env & env, 
                              Pds::Damage damage);
  void overwriteDamage(long index, 
                       const PSEvt::EventKey &eventKey, 
                       boost::shared_ptr<PSEvt::EventId> eventId, 
                       Pds::Damage damage);
  void close();
  boost::shared_ptr<Translator::HdfWriterFromEvent> hdfWriter() { return m_hdfWriter; };
  const PSEvt::EventKey & eventKey() const { return m_eventKey; }
  bool arrayTypeDatasetsCreated() const { return m_datasetsCreated == ArrayForTypeTimeDamage; }
  bool arrayDatasetsCreated() const { return m_datasetsCreated == ArrayForTypeTimeDamage or 
      m_datasetsCreated == ArrayForOnlyTimeDamage; }

 private:
  PSEvt::EventKey m_eventKey;
  hdf5pp::Group m_group;

  boost::shared_ptr<Translator::HdfWriterFromEvent> m_hdfWriter;
  boost::shared_ptr<Translator::HdfWriterEventId> m_hdfWriterEventId;
  boost::shared_ptr<Translator::HdfWriterDamage> m_hdfWriterDamage;

  bool m_writtenForThisEvent;

  //  If the first entries for a type are blanks we cannot store them until we see
  //  actual data and can create the datasets.  So we keep track of how many initial 
  //  blanks to write once we have the datasets.  We will write entries for time and
  //  damage when initial blanks come in.  This is the only point when the lengths of
  //  the time/damage datasest is not equal to that of the type dataset.
  size_t m_initialBlanks; 
  size_t m_totalEntries; // total entries counts initial blanks that we have not been
                         // able to store yet. It will be equal to m_initialBlanks if
                         // m_initialBlanks is nonzero.

  typedef enum {None=0, ScalarForType=1, ArrayForTypeTimeDamage=2, 
                ArrayForOnlyTimeDamage=3} DatasetsCreated;
  std::string datasetsCreatedStr();
  DatasetsCreated m_datasetsCreated;
};

typedef std::pair<Pds::Src, std::string> SrcKeyPair;

inline SrcKeyPair getSrcKeyPair(const PSEvt::EventKey &eventKey) {
  SrcKeyPair srcKeyPair;
  srcKeyPair.first = eventKey.src();
  srcKeyPair.second = eventKey.key();
  return srcKeyPair;
}

/**
 * comparison operator for Src's.
 * It is important that this be coordinated to some extent with H5GroupNames::srcName.
 * For instance, if all Control level sources get the same name, then we need all 
 * two Control level sources to be equal to one another in this comparison.
 */
class LessSrcKeyPair {
 public:
 LessSrcKeyPair(const std::string &calibKey) : m_calibKey(calibKey) {} 
  bool operator()(const SrcKeyPair &a, const SrcKeyPair &b) {
    const Pds::Src & aSrc = a.first;
    const Pds::Src & bSrc = b.first;
    if ((aSrc.level() ==  Pds::Level::Event) and (bSrc.level() ==  Pds::Level::Event)) return false;
    if ((aSrc.level() ==  Pds::Level::Control) and (bSrc.level() ==  Pds::Level::Control)) return false;    
    if (aSrc < bSrc) return true;
    if (bSrc < aSrc) return false;
    
    const std::string *aStr = & a.second;
    const std::string *bStr = & b.second;
    if (a.second == m_calibKey) aStr = & EMPTY_STRING;
    if (b.second == m_calibKey) bStr = & EMPTY_STRING;
    return (*aStr < *bStr);
  }
  private:
      std::string m_calibKey;
      static const std::string EMPTY_STRING;
};
      
typedef std::map< SrcKeyPair, SrcKeyGroup, LessSrcKeyPair > SrcKeyMap;

/**
 *  @ingroup Translator
 *
 *  @brief class to manage the src groups in a type group.
 *
 *  Instances of TypeGroup are created and managed by TypeSrcKeyH5GroupDirectory,
 *  refer to that class for the top level interface for translated data to the hdf5
 *  file.
 * 
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class TypeGroup {
 public:
 TypeGroup(const std::string & calibrationKey) : 
  m_srcKeyMap(LessSrcKeyPair(calibrationKey)) {};
 TypeGroup(hdf5pp::Group &group, 
           boost::shared_ptr<Translator::HdfWriterEventId> hdfWriterEventId,
           boost::shared_ptr<Translator::HdfWriterDamage> hdfWriterDamage,
           const std::string &calibrationKey) 
   : m_group(group), 
    m_srcKeyMap(LessSrcKeyPair(calibrationKey)),
    m_hdfWriterEventId(hdfWriterEventId),
    m_hdfWriterDamage(hdfWriterDamage) {};
    
  SrcKeyMap & srcKeyMap() { return m_srcKeyMap; };
  hdf5pp::Group group() { return m_group; };
 private:
  hdf5pp::Group m_group;
  SrcKeyMap m_srcKeyMap;
  boost::shared_ptr<Translator::HdfWriterEventId> m_hdfWriterEventId;
  boost::shared_ptr<Translator::HdfWriterDamage> m_hdfWriterDamage;
};

 typedef std::map<const std::string, TypeGroup> TypeMapContainer;

/**
 *  @ingroup Translator
 *
 *  @brief class to manage the type and source groups in the hdf5 file.
 * 
 * This class manages the type and source groups that are children to the
 * configure and calib cycle groups in the hdf5 file. It provides the top 
 * level interface for the translator module for working with these groups.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class TypeSrcKeyH5GroupDirectory {
 public:
  TypeSrcKeyH5GroupDirectory() {}
  void setH5GroupNames(boost::shared_ptr<Translator::H5GroupNames> h5GroupNames) {
    m_h5GroupNames = h5GroupNames;
  }
  void setAliasMap(boost::shared_ptr<PSEvt::AliasMap> aliasMap) { m_aliasMap = aliasMap; }
  std::string getAlias(const Pds::Src &src);
  void setEventIdAndDamageWriters(boost::shared_ptr<Translator::HdfWriterEventId> hdfWriterEventId,
                                  boost::shared_ptr<Translator::HdfWriterDamage> hdfWriterDamage) 
  {
    m_hdfWriterEventId = hdfWriterEventId; 
    m_hdfWriterDamage = hdfWriterDamage;
  }

  void closeGroups(); 
  void clearMaps();
  void markAllSrcKeyGroupsNotWrittenForEvent();
  TypeMapContainer::iterator findType(const PSEvt::EventKey &eventKey);
  TypeMapContainer::iterator beginType();
  TypeMapContainer::iterator endType();
  TypeGroup & addTypeGroup(const PSEvt::EventKey &eventKey,
                           hdf5pp::Group & parentGroup);
  SrcKeyMap::iterator findSrcKey(const PSEvt::EventKey &eventKey);
  SrcKeyMap::iterator endSrcKey(const PSEvt::EventKey &eventKey);
  SrcKeyGroup & addSrcKeyGroup(const PSEvt::EventKey &eventKey, 
                               boost::shared_ptr<Translator::HdfWriterFromEvent> hdfWriter);
  /* partitions all the eventKeys that are in the TypeSrcKey directory into three sets, 
     based on a input set of srcs. The sets are:
       outputSrcMap -            keys that have not been written during the present event, and
                                 have a src in the srcs argument
       outputOtherNotWritten -   keys that have not been written during the present event, and 
                                 do not have a src in the srcs argument
       outputWrittenKeys -       Any key that has already been written during the present event
                                 regardless of whether or not in the srcs argument
   */
  void getNotWrittenSrcPartition(const std::set<Pds::Src> & srcs, 
                                 std::map<Pds::Src, std::vector<PSEvt::EventKey> > & outputSrcMap, 
                                 std::vector<PSEvt::EventKey> & outputOtherNotWritten,
                                 std::vector<PSEvt::EventKey> & outputWrittenKeys);
  void dump();
 private:
  TypeMapContainer m_map;
  boost::shared_ptr<Translator::HdfWriterEventId> m_hdfWriterEventId;
  boost::shared_ptr<Translator::HdfWriterDamage> m_hdfWriterDamage;
  boost::shared_ptr<Translator::H5GroupNames> m_h5GroupNames;
  boost::shared_ptr<PSEvt::AliasMap> m_aliasMap;
};

} // namespace Translator
#endif
