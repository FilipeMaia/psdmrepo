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
#include "PSEvt/SrcCmp.h"
#include "PSEvt/TypeInfoUtils.h"
#include "PSEvt/Event.h"
#include "PSEnv/Env.h"

#include "hdf5pp/Group.h"

#include "Translator/HdfWriterBase.h"
#include "Translator/HdfWriterEventId.h"
#include "Translator/HdfWriterDamage.h"
#include "Translator/DataSetCreationProperties.h"

namespace Translator {

class LessSrc {
 public:
  bool operator()(const Pds::Src& lhs, const Pds::Src& rhs) const { return PSEvt::SrcCmp::cmp(lhs,rhs); }
};


class SrcKeyGroup {
 public:
 SrcKeyGroup() : 
    m_writtenForThisEvent(false),
    m_initialBlanks(0),
    m_totalEntries(0),
    m_datasetsCreated(None) {};

  SrcKeyGroup(hdf5pp::Group group, 
              const PSEvt::EventKey &eventKey,
              boost::shared_ptr<Translator::HdfWriterBase> hdfWriter, 
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
  boost::shared_ptr<Translator::HdfWriterBase> hdfWriter() { return m_hdfWriter; };
  const PSEvt::EventKey & eventKey() { return m_eventKey; }
  bool arrayTypeDatasetsCreated() { return m_datasetsCreated == ArrayForTypeTimeDamage; }

 private:
  PSEvt::EventKey m_eventKey;
  hdf5pp::Group m_group;

  boost::shared_ptr<Translator::HdfWriterBase> m_hdfWriter;
  boost::shared_ptr<Translator::HdfWriterEventId> m_hdfWriterEventId;
  boost::shared_ptr<Translator::HdfWriterDamage> m_hdfWriterDamage;

  bool m_writtenForThisEvent;

  //  If the first entries for a type are blanks we cannot store them until we see
  //  actual data and can create the datasets.  So we keep track of how many initial 
  //  blanks to write once we have the datasets.  We will write entries for time and
  //  damage when initial blanks come in.  This is the only point when then lengths of
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

class LessSrcKeyPair {
 public:
  bool operator()(const SrcKeyPair &a, const SrcKeyPair &b) {
    const Pds::Src & aSrc = a.first;
    const Pds::Src & bSrc = b.first;
    int src_diff = PSEvt::SrcCmp::cmp(aSrc, bSrc);
    if (src_diff<0) return true;
    if (src_diff>0) return false;
    const std::string & aStr = a.second;
    const std::string & bStr = b.second;
      return (aStr < bStr);
  }
};
      
typedef std::map< SrcKeyPair, SrcKeyGroup, LessSrcKeyPair > SrcKeyMap;

class TypeGroup {
 public:
  TypeGroup() {};
 TypeGroup(hdf5pp::Group &group, 
           boost::shared_ptr<Translator::HdfWriterEventId> hdfWriterEventId,
           boost::shared_ptr<Translator::HdfWriterDamage> hdfWriterDamage) 
   : m_group(group),
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

 typedef std::map<const std::type_info *, TypeGroup, PSEvt::TypeInfoUtils::lessTypeInfoPtr> TypeMapContainer;

class TypeSrcKeyH5GroupDirectory {
 public:
  TypeSrcKeyH5GroupDirectory() {};
  void setEventIdAndDamageWriters(boost::shared_ptr<Translator::HdfWriterEventId> hdfWriterEventId,
                                  boost::shared_ptr<Translator::HdfWriterDamage> hdfWriterDamage) 
  {
    m_hdfWriterEventId = hdfWriterEventId; 
    m_hdfWriterDamage = hdfWriterDamage;
  }

  void closeGroups(); 
  void clearMaps();
  void markAllSrcKeyGroupsNotWrittenForEvent();
  TypeMapContainer::iterator findType(const std::type_info *);
  TypeMapContainer::iterator endType();
  TypeGroup & addTypeGroup(const std::type_info *, 
                           hdf5pp::Group & parentGroup,
                           bool short_bld_name);
  SrcKeyMap::iterator findSrcKey(const PSEvt::EventKey &);
  SrcKeyMap::iterator endSrcKey(const std::type_info *);
  SrcKeyGroup & addSrcKeyGroup(const PSEvt::EventKey &eventKey, 
                               boost::shared_ptr<Translator::HdfWriterBase> hdfWriter);
  void getNotWrittenSrcPartition(const std::set<Pds::Src, LessSrc> & srcs, 
                                 std::map<Pds::Src, std::vector<PSEvt::EventKey>, LessSrc > & outputSrcMap, 
                                 std::vector<PSEvt::EventKey> & outputOtherNotWritten,
                                 std::vector<PSEvt::EventKey> & outputWrittenKeys);
 private:
  TypeMapContainer m_map;
  boost::shared_ptr<Translator::HdfWriterEventId> m_hdfWriterEventId;
  boost::shared_ptr<Translator::HdfWriterDamage> m_hdfWriterDamage;
};

} // namespace Translator
#endif
