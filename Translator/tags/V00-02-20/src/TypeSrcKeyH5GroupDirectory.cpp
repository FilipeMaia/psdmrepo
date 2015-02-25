#include <set>
#include <string>
#include <sstream>

#include "ErrSvc/Issue.h"
#include "MsgLogger/MsgLogger.h"
#include "psddl_hdf2psana/SchemaConstants.h"
#include "Translator/TypeSrcKeyH5GroupDirectory.h"
#include "Translator/H5GroupNames.h"
#include "Translator/hdf5util.h"

using namespace std;
using namespace Translator;

namespace {
const char* logger = "TypeSrcKeyH5GroupDirectory"; 

}; // end the local namespace

const std::string LessSrcKeyPair::EMPTY_STRING;

/////////////////////////////////////////////////////////
void SrcKeyGroup::make_timeDamageDatasets()
{
  if (m_hdfWriterEventId) m_hdfWriterEventId->make_dataset(group());
  if (m_hdfWriterDamage) m_hdfWriterDamage->make_datasets(group());
  m_datasetsCreated = ArrayForOnlyTimeDamage;
}

void SrcKeyGroup::make_typeDatasets(DataTypeLoc dataTypeLoc,
                                    PSEvt::Event &evt, PSEnv::Env &env, 
                                    const DataSetCreationProperties & dsetCreateProp) {
  if (m_datasetsCreated != ArrayForOnlyTimeDamage) {
    MsgLog(logger,fatal,"make_typeDatasets called but datasetCreated is not arrayForOnlyTimeDamage, call make_timeDamageDatasets first");
  }

  m_hdfWriter->make_datasets(dataTypeLoc, m_group, m_eventKey, evt, env, 
                             dsetCreateProp.shuffle(), 
                             dsetCreateProp.deflate(),
                             dsetCreateProp.chunkPolicy());


  m_datasetsCreated = ArrayForTypeTimeDamage;

  while (m_initialBlanks > 0) {
    m_hdfWriter->addBlank(m_group);
    --m_initialBlanks;
  }
}

void SrcKeyGroup::make_datasets(DataTypeLoc dataTypeLoc,
                                PSEvt::Event &evt, PSEnv::Env &env, 
                                const DataSetCreationProperties & dsetCreateProp) {
  if (m_datasetsCreated != None) {
    MsgLog(logger,fatal,"make_datasets called but datasetCreated is not None");
  }
  m_hdfWriter->make_datasets(dataTypeLoc, m_group, m_eventKey, evt, env, 
                             dsetCreateProp.shuffle(), 
                             dsetCreateProp.deflate(),
                             dsetCreateProp.chunkPolicy());  

  if (m_hdfWriterEventId) m_hdfWriterEventId->make_dataset(m_group);
  if (m_hdfWriterDamage) m_hdfWriterDamage->make_datasets(m_group);

  m_datasetsCreated = ArrayForTypeTimeDamage;

  if (m_initialBlanks > 0) MsgLog(logger, fatal, "make_datasets called but there are initial blanks.");
}

void SrcKeyGroup::storeData(const PSEvt::EventKey & eventKey, DataTypeLoc dataTypeLoc,
                            PSEvt::Event &evt, PSEnv::Env &env) {
  if (m_datasetsCreated == None) {
    m_hdfWriter->store(dataTypeLoc, group(), eventKey, evt, env);
    m_datasetsCreated = ScalarForType;
    m_totalEntries = 1;
  } else {
    ostringstream msg;
    const Pds::Src & src = eventKey.src();
    msg << "Cannot make a scalar dataset, datasetsCreated is: " 
        << datasetsCreatedStr() << " eventKey = " << eventKey
        << " full Src: " << std::hex << " log=0x" << src.log()
        << " phy=0x" << src.phy();
    throw ErrSvc::Issue(ERR_LOC, msg.str());
  }
}

long SrcKeyGroup::appendDataTimeAndDamage(const PSEvt::EventKey & eventKey, 
                                          DataTypeLoc dataTypeLoc,
                                          PSEvt::Event &evt, 
                                          PSEnv::Env &env, 
                                          boost::shared_ptr<PSEvt::EventId> eventId,
                                          Pds::Damage damage) {
  if (m_datasetsCreated != ArrayForTypeTimeDamage) {
    MsgLog(logger,fatal,"appending data but type dataset not created");
  }
  m_hdfWriter->append(dataTypeLoc, group(), eventKey, evt, env);
  if (m_hdfWriterEventId) m_hdfWriterEventId->append(group(), *eventId);
  if (m_hdfWriterDamage) m_hdfWriterDamage->append(group(), damage, HdfWriterDamage::ValidEntry);
  ++m_totalEntries;
  return m_totalEntries-1;
}

long SrcKeyGroup::appendBlankTimeAndDamage(const PSEvt::EventKey & eventKey, 
                                           boost::shared_ptr<PSEvt::EventId> eventId,
                                           Pds::Damage damage) {
  if (m_datasetsCreated == None or m_datasetsCreated == ScalarForType) {
    ostringstream msg;
    msg << "cannot append blank, datasetsCreated is " 
        << datasetsCreatedStr() << " eventKey=" << eventKey
        << " eventId=" << *eventId << " damage=" << std::hex << damage.value();
    throw std::runtime_error(msg.str());
  }
  if (m_datasetsCreated == ArrayForTypeTimeDamage) {
    m_hdfWriter->addBlank(group());
  } else {
    ++m_initialBlanks;
    MsgLog(logger,trace,"set initial blank count to " << m_initialBlanks);
  }
  if (m_hdfWriterEventId) m_hdfWriterEventId->append(group(), *eventId);
  if (m_hdfWriterDamage) {
    m_hdfWriterDamage->append(group(), damage, HdfWriterDamage::BlankEntry);
  }
  ++m_totalEntries;
  MsgLog(logger,trace,"appended time, damage of " << damage.value() << " total entries=" << m_totalEntries);
  return m_totalEntries-1;
}

void SrcKeyGroup::overwriteDataAndDamage(long index, 
                                         const PSEvt::EventKey &eventKey, 
                                         DataTypeLoc dataTypeLoc,
                                         PSEvt::Event & evt, 
                                         PSEnv::Env & env, 
                                         Pds::Damage damage) {
  MsgLog(logger,trace,"overwriteDataAndDamage: index=" << index );
  m_hdfWriter->store_at(dataTypeLoc, index, group(), eventKey, evt, env);
  if (m_hdfWriterDamage) {
    m_hdfWriterDamage->store_at(index, group(), damage, HdfWriterDamage::ValidEntry);
  }
}

void SrcKeyGroup::overwriteDamage(long index, 
                                  const PSEvt::EventKey &eventKey, 
                                  boost::shared_ptr<PSEvt::EventId> eventId, 
                                  Pds::Damage damage) {
  MsgLog(logger,trace,"overwriteDamage: index=" << index );
  if (m_hdfWriterDamage) {
    m_hdfWriterDamage->store_at(index, group(), damage, HdfWriterDamage::BlankEntry);
  } else {
    MsgLog(logger,warning, "overwriteDamage called but no damage writer is NULL");
  }
}


void SrcKeyGroup::close()  {
  if (m_datasetsCreated == ArrayForTypeTimeDamage or 
      m_datasetsCreated == ArrayForOnlyTimeDamage) {
    if (m_hdfWriterEventId) m_hdfWriterEventId->closeDataset(m_group);
    if (m_hdfWriterDamage) m_hdfWriterDamage->closeDatasets(m_group);
  } 
  if (m_datasetsCreated == ArrayForTypeTimeDamage) {
    m_hdfWriter->closeDatasets(m_group);
  }
  m_group.close(); 
};

string SrcKeyGroup::datasetsCreatedStr() {
  switch (m_datasetsCreated) {
  case None:
    return "None";
  case ScalarForType:
    return "ScalarForType";
  case ArrayForTypeTimeDamage:
    return "ArrayForTypeTimeDamage";
  case ArrayForOnlyTimeDamage:
    return "ArrayForOnlyTimeDamage";
  }
  return "*undefined*";
}

/////////////////////////////////////////////////
void TypeSrcKeyH5GroupDirectory::closeGroups() { 
  TypeMapContainer::iterator typeIter;
  for (typeIter= beginType(); typeIter != endType(); ++typeIter) {
    TypeGroup & typeGroup = typeIter->second;
    SrcKeyMap & srcKeyMap = typeGroup.srcKeyMap();
    SrcKeyMap::iterator srcIter;
    for (srcIter = srcKeyMap.begin(); srcIter != srcKeyMap.end(); ++srcIter) {
      SrcKeyGroup & srcKeyGroup = srcIter->second;
      srcKeyGroup.close();
    }
    typeGroup.group().close();
  }
}

void TypeSrcKeyH5GroupDirectory::clearMaps() {
  TypeMapContainer::iterator typeIter;
  for (typeIter= beginType(); typeIter != endType(); ++typeIter) {
    TypeGroup & typeGroup = typeIter->second;
    SrcKeyMap & srcKeyMap = typeGroup.srcKeyMap();
    srcKeyMap.clear();
  }
  m_map.clear();
}


void TypeSrcKeyH5GroupDirectory::markAllSrcKeyGroupsNotWrittenForEvent() {
  TypeMapContainer::iterator typeIter;
  for (typeIter= beginType(); typeIter != endType(); ++typeIter) {
    TypeGroup & typeGroup = typeIter->second;
    SrcKeyMap & srcKeyMap = typeGroup.srcKeyMap();
    SrcKeyMap::iterator srcIter;
    for (srcIter = srcKeyMap.begin(); srcIter != srcKeyMap.end(); ++srcIter) {
      SrcKeyGroup & srcKeyGroup = srcIter->second;
      srcKeyGroup.written(false);
    }
  }
}


TypeMapContainer::iterator TypeSrcKeyH5GroupDirectory::findType(const PSEvt::EventKey &eventKey) {
  const std::type_info * typeInfoPtr = eventKey.typeinfo();
  const std::string &key = eventKey.key();
  std::string h5GroupName = m_h5GroupNames->nameForType(typeInfoPtr,key);
  return m_map.find(h5GroupName);
}

TypeMapContainer::iterator TypeSrcKeyH5GroupDirectory::beginType() {
  return m_map.begin();
}

TypeMapContainer::iterator TypeSrcKeyH5GroupDirectory::endType() {
  return m_map.end();
}

TypeGroup & TypeSrcKeyH5GroupDirectory::addTypeGroup(const PSEvt::EventKey &eventKey, hdf5pp::Group & parentGroup) {
  const std::type_info * typeInfoPtr = eventKey.typeinfo();
  const std::string &key = eventKey.key();
  string groupName = m_h5GroupNames->nameForType(typeInfoPtr, key);
  hdf5pp::Group group = parentGroup.createGroup(groupName);
  m_h5GroupNames->addTypeAttributes(group, typeInfoPtr, key);
  TypeMapContainer::iterator pos = m_map.find(groupName);
  if (pos != m_map.end()) MsgLog(logger, error, "addTypeGroup is overwriting existing entry for " << groupName);
  TypeMapContainer::value_type newElem(groupName, TypeGroup(group,
                                                            m_hdfWriterEventId,
                                                            m_hdfWriterDamage,
                                                            m_h5GroupNames->calibratedKey()));
  m_map.insert(newElem);
  return m_map.find(newElem.first)->second;
}

SrcKeyMap::iterator TypeSrcKeyH5GroupDirectory::findSrcKey(const PSEvt::EventKey &eventKey) {
  const type_info * typeInfoPtr = eventKey.typeinfo();
  TypeMapContainer::iterator typePos = findType(eventKey);
  if (typePos == endType()) MsgLog(logger,fatal,"findSrc - type_info " << PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr) << " not stored");
  TypeGroup & typeGroup = typePos->second;
  SrcKeyMap & srcKeyMap = typeGroup.srcKeyMap();
  const Pds::Src & src = eventKey.src();
  const string & key = eventKey.key();
  SrcKeyPair srcStrPair = make_pair(src,key);
  SrcKeyMap::iterator srcPos = srcKeyMap.find(srcStrPair);
  return srcPos;
}

SrcKeyMap::iterator TypeSrcKeyH5GroupDirectory::endSrcKey(const PSEvt::EventKey &eventKey) {
  TypeMapContainer::iterator typePos = findType(eventKey);
  if (typePos == endType()) MsgLog(logger,fatal,"endSrc - typeInfo " << PSEvt::TypeInfoUtils::typeInfoRealName(eventKey.typeinfo()) << " not stored");
  TypeGroup & typeGroup = typePos->second;
  SrcKeyMap &srcKeyMap = typeGroup.srcKeyMap();
  return srcKeyMap.end();
}

string TypeSrcKeyH5GroupDirectory::getAlias(const Pds::Src &src) {
  if (m_aliasMap) {
    return m_aliasMap->alias(src);
  }
  return "";
}

SrcKeyGroup & TypeSrcKeyH5GroupDirectory::addSrcKeyGroup(const PSEvt::EventKey &eventKey, 
                                                         boost::shared_ptr<Translator::HdfWriterFromEvent> hdfWriter) {
  TypeMapContainer::iterator typePos = findType(eventKey);
  if (typePos == endType()) MsgLog(logger,fatal,"addSrcKeyGroup " << eventKey << " not stored");
  TypeGroup & typeGroup = typePos->second;
  SrcKeyMap &srcKeyMap = typeGroup.srcKeyMap();
  const Pds::Src &src = eventKey.src();
  const string &key = eventKey.key();
  SrcKeyPair srcStrPair = make_pair(src,key);
  pair<string,string> nameRes = m_h5GroupNames->nameForSrcKey(src,key);
  string srcKeyGroupName = nameRes.first;
  string keyInName = nameRes.second;
  hdf5pp::Group typeH5Group = typeGroup.group();
  hdf5pp::Group srcH5Group = typeH5Group.createGroup(srcKeyGroupName);
  uint64_t srcVal = (uint64_t(src.phy()) << 32) + src.log();
  srcH5Group.createAttr<uint64_t>(psddl_hdf2psana::srcAttrName).store(srcVal);
  srcH5Group.createAttr<const char *>(psddl_hdf2psana::eventKeyAttrName).store(key.c_str());
  srcH5Group.createAttr<const char *>(psddl_hdf2psana::h5GroupNameKeyAttrName).store(keyInName.c_str());
  MsgLog(logger,trace,"addSrcKeyGroup " << srcKeyGroupName);
  return (srcKeyMap[ srcStrPair ] = SrcKeyGroup(srcH5Group,
                                                eventKey,
                                                hdfWriter,
                                                m_hdfWriterEventId,
                                                m_hdfWriterDamage));
}

void TypeSrcKeyH5GroupDirectory::getNotWrittenSrcPartition(const set<Pds::Src> & srcs, 
                                                           map<Pds::Src, vector<PSEvt::EventKey> > & outputSrcMap, 
                                                           vector<PSEvt::EventKey> & outputOtherNotWritten,                          
                                                           vector<PSEvt::EventKey> & outputWrittenKeys)
{
  TypeMapContainer::iterator typePos;
  for (typePos = beginType(); typePos != endType(); ++typePos) {
    TypeGroup & typeGroup = typePos->second;
    SrcKeyMap & srcKeyMap = typeGroup.srcKeyMap();
    SrcKeyMap::iterator srcKeyPos;
    for (srcKeyPos = srcKeyMap.begin(); srcKeyPos != srcKeyMap.end(); ++srcKeyPos) {
      SrcKeyGroup & srcKeyGroup = srcKeyPos->second;
      if (srcKeyGroup.written()) {
        outputWrittenKeys.push_back(srcKeyGroup.eventKey());
        continue;
      }
      const SrcKeyPair & srcKeyPair = srcKeyPos->first;
      const Pds::Src & src = srcKeyPair.first;
      // we don't care about the key here, just the src
      set<Pds::Src>::iterator partPos = srcs.find(src);
      bool srcInPartition = partPos != srcs.end();
      if (not srcInPartition) {
        outputOtherNotWritten.push_back(srcKeyGroup.eventKey());
        continue;
      } 
      vector<PSEvt::EventKey> & srcNotWrittenList = outputSrcMap[src];
      srcNotWrittenList.push_back(srcKeyGroup.eventKey());
    }
  }
}

void TypeSrcKeyH5GroupDirectory::dump() {
  ostringstream msg;
  msg << " ** TypeSrcKeyH5GroupDirectory::dump ** " << endl;
  TypeMapContainer::iterator typePos;
  for (typePos = beginType(); typePos != endType(); ++typePos) {
    const std::string typeGroupName = typePos->first;
    TypeGroup & typeGroup = typePos->second;
    msg << typeGroupName << " : " << endl;
    SrcKeyMap & srcKeyMap = typeGroup.srcKeyMap();
    SrcKeyMap::iterator srcPos;
    for (srcPos = srcKeyMap.begin(); srcPos != srcKeyMap.end(); ++srcPos) {
      const SrcKeyPair & srcKeyPair = srcPos->first;
      const Pds::Src & src = srcKeyPair.first;
      const string & key = srcKeyPair.second;
      pair<string,string> nameRes = m_h5GroupNames->nameForSrcKey(src,key);
      string groupName = nameRes.first;
      msg << "  srckey name = " <<   groupName
          << " log=0x" << std::hex << src.log() << " phy=0x" << src.phy()
          << " eventKey='" << key << "'" << endl;
    }
  }
  MsgLog(logger,info,msg.str());
}
