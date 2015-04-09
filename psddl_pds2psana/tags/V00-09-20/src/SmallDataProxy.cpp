//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class SmallDataProxy
//
// Author List:
//     David Schneider
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_pds2psana/SmallDataProxy.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <sstream>
#include "boost/filesystem.hpp"
#include "boost/algorithm/string/predicate.hpp"
#include <sys/stat.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Exceptions.h"
#include "PSEvt/ProxyI.h"
#include "PSEvt/ProxyDict.h"
#include "PSEvt/ProxyDictHist.h"
#include "PSEvt/TypeInfoUtils.h"
#include "PSEvt/Exceptions.h"
#include "psddl_pds2psana/smldata.ddl.h"
#include "psddl_pds2psana/dispatch.h"
#include "pdsdata/compress/CompressedXtc.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psddl_pds2psana;

#define DEBUGMSG debug
#define TRACEMSG trace

namespace {
  
  const char * logger = "SmallDataProxy";

  boost::shared_ptr<Pds::Xtc> uncompressIfNeeded(boost::shared_ptr<Pds::Xtc> xtc) {
    if (xtc->contains.compressed()) {
      return Pds::CompressedXtc::uncompress(*xtc);
    }
    return xtc;
  }

class LargeObjectProxy : public PSEvt::ProxyI, public boost::enable_shared_from_this<LargeObjectProxy> {
  public:
  static boost::shared_ptr<LargeObjectProxy> makeLargeObjectProxy(const std::type_info *evKeyType,
                                                                  SmallDataProxy::ObjId objId,
                                                                  boost::shared_ptr<SmallDataProxy> parentProxy) {
    return boost::shared_ptr<LargeObjectProxy>( new LargeObjectProxy(evKeyType, objId, parentProxy));
  }

  protected:
  LargeObjectProxy(const std::type_info *evKeyType,
                   SmallDataProxy::ObjId objId,
                   boost::shared_ptr<SmallDataProxy> parentProxy) :
    m_evKeyType(evKeyType)
    , m_objId(objId)
    , m_parentProxy(parentProxy) 
  {
    MsgLog(logger, DEBUGMSG, "LargeObjectProxy created for " 
           << PSEvt::TypeInfoUtils::typeInfoRealName(evKeyType)
           << " objId=" << objId);
  }

  virtual boost::shared_ptr<void> getImpl(PSEvt::ProxyDictI* dict,
                                          const Pds::Src& source, 
                                          const std::string& key)
  {
    if (not m_parentProxy->isFinalized()) {
      MsgLog(logger, error, "parentProxy for event is not finalized");
      return boost::shared_ptr<void>();
    }
    if (m_cachedPointerToReturn) return m_cachedPointerToReturn;

    PSEvt::EventKey evKey(m_evKeyType, source, key);
    MsgLog(logger, DEBUGMSG, "LargeObjectProxy calling parentProxy.get for " 
           << evKey << " objId=" << m_objId);
    // we save a reference to ourselves before calling parent proxy because there
    // is a small chance the last reference to this proxy gets deleted before we
    // return. This is because for items proxied in the configStore, the parentProxy
    // will replace this objects entry in the configStore.
    boost::shared_ptr<LargeObjectProxy> referenceToPreventDeleteBeforeReturn = this->shared_from_this();
    m_cachedPointerToReturn = m_parentProxy->get(m_objId, evKey);
    return m_cachedPointerToReturn;
  }

  private:
  const std::type_info *m_evKeyType;
  SmallDataProxy::ObjId m_objId;
  boost::shared_ptr<SmallDataProxy> m_parentProxy;
  boost::shared_ptr<void>  m_cachedPointerToReturn;
};

}; // local namespace


//             ----------------------------------------
//             -- Public Function Member Definitions --
//             ----------------------------------------

namespace psddl_pds2psana {

//----------------
// Factory function to create object
//----------------

boost::shared_ptr<SmallDataProxy> 
SmallDataProxy::makeSmallDataProxy(const XtcInput::XtcFileName & smallXtcFileName, bool liveMode, 
                                   unsigned liveTimeOut, XtcConverter &cvt,
                                   PSEvt::Event* evt, PSEnv::Env &env, unsigned optimalReadBytes)
{
  boost::shared_ptr<SmallDataProxy> toReturn;
  if (not smallXtcFileName.small()) {
    // return null pointer if this is not a small xtc file
    return toReturn;
  }
  boost::shared_ptr<PSEvt::Event> evtPtr;
  if (evt) evtPtr = evt->shared_from_this();
  boost::shared_ptr<PSEnv::Env> envPtr = env.shared_from_this();
  SmallDataProxy *rawPointer = new SmallDataProxy(smallXtcFileName, liveMode, liveTimeOut, cvt, 
                                                  evtPtr, envPtr, optimalReadBytes);
  toReturn = boost::shared_ptr<SmallDataProxy>(rawPointer);
  return toReturn;
}


//----------------
// Constructors --
//----------------

SmallDataProxy::SmallDataProxy(const XtcInput::XtcFileName &smallXtcFileName, bool liveMode,
                               unsigned liveTimeOut, XtcConverter &cvt,
                               boost::shared_ptr<PSEvt::Event> evt, 
                               boost::shared_ptr<PSEnv::Env> env, 
                               unsigned optimalReadBytes)

  : m_smallXtcFileName(smallXtcFileName)
  , m_liveMode(liveMode)
  , m_liveTimeOut(liveTimeOut)
  , m_cvt(cvt)
  , m_finalized(false)
  , m_optimalReadBytes(optimalReadBytes)
  , m_triedToOpen(false)
  , m_evtForProxies(boost::weak_ptr<PSEvt::Event>(evt))
  , m_nextObjId(1)
{   
  if (not m_smallXtcFileName.small()) {
    MsgLog(logger, fatal, "small xtc file" << m_smallXtcFileName << " is not small");
    return;
  }
  boost::filesystem::path smallDataDir = 
    boost::filesystem::path(m_smallXtcFileName.path()).parent_path();
  if (not boost::algorithm::ends_with(smallDataDir.string(), std::string("smalldata"))) {
    MsgLog(logger, fatal, "small data file in not in a directory called "
           << "'smalldata', path is: " << m_smallXtcFileName);
    return;
  }
  if (m_liveMode and (m_liveTimeOut == 0)) {
    MsgLog(logger, warning, "live mode identified but liveTimeOut is 0, setting liveMode to False");
    m_liveMode = false;
  }
  std::string largeBaseName = m_smallXtcFileName.largeBasename();
  if (m_liveMode and (not boost::algorithm::ends_with(largeBaseName, ".inprogress"))) {
    // in live mode, the small and large files may be arriving at different times. We want to first
    // check for an inprogress file
    largeBaseName += ".inprogress";
  }
  boost::filesystem::path largeDataDir = smallDataDir.parent_path();
  boost::filesystem::path largeDataPath = largeDataDir / largeBaseName;
  m_largeXtcFileName = XtcInput::XtcFileName(largeDataPath.string());

  m_configStoreForProxies = boost::weak_ptr<PSEnv::EnvObjectStore>(env->configStore().shared_from_this());
  boost::shared_ptr<PSEvt::AliasMap> aliasMap = env->aliasMap();
  m_evtForLargeData.reset(new PSEvt::Event(boost::make_shared<PSEvt::ProxyDict>(aliasMap)));
  boost::shared_ptr<PSEvt::ProxyDictHist> proxyDictHist = boost::make_shared<PSEvt::ProxyDictHist>(aliasMap);
}

SmallDataProxy::~SmallDataProxy() 
{
}

bool SmallDataProxy::isSmallDataProxy(const Pds::TypeId &typeId) {
  return (typeId.id() == Pds::TypeId::Id_SmlDataProxy);
}

const Pds::TypeId SmallDataProxy::getSmallDataProxiedType(const Pds::Xtc * origXtc) {
  if (not origXtc) MsgLog(logger, fatal, "getSmallDataProxiedType passed NULL pointer");
  if (not isSmallDataProxy(origXtc->contains)) {
    MsgLog(logger, fatal, "Internal error: xtc typeid is not SmlDataProxy");
  }
  if (1 != origXtc->contains.compressed_version()) {
    MsgLog(logger, fatal, "version is not 1 for smallDataProxy. Version=" << origXtc->contains.compressed_version());
  }
    
  // deal with uncompressing from a raw pointer
  boost::shared_ptr<Pds::Xtc> uXtc;  // make sure uXtc is around 
  const Pds::Xtc *xtc = 0;           // as long as xtc is
  if (origXtc->contains.compressed()) {
    uXtc = Pds::CompressedXtc::uncompress(*origXtc);
    xtc = uXtc.get();
  } else {
    xtc = origXtc;
  }
  
  const Pds::SmlData::ProxyV1 * smlDataProxy = static_cast<Pds::SmlData::ProxyV1 *>(static_cast<void *>(xtc->payload()));
  return smlDataProxy->type();
}

std::vector<const std::type_info *> 
SmallDataProxy::getSmallConvertTypeInfoPtrs(const Pds::Xtc * xtc, 
                                            const XtcConverter &cvt) {
  const Pds::TypeId proxiedTypeId = getSmallDataProxiedType(xtc);
  return cvt.getConvertTypeInfoPtrs(proxiedTypeId);
}

bool 
SmallDataProxy::addProxyChecksFailed(const boost::shared_ptr<Pds::Xtc>& xtc, 
                                     XtcConverter &cvt) 
{
  if (not xtc) {
    MsgLog(logger, error, "xtc pointer is null");
    return true;
  }
  
  // protect from zero-payload data
  if (xtc->sizeofPayload() <= 0) {
    MsgLog(logger, error, "0 size xtc smallDataProxy object");
    return true;
  }
  
  const Pds::TypeId& smallProxyTypeId = xtc->contains;
  
  if (not isSmallDataProxy(smallProxyTypeId)) {
    MsgLog(logger, error, "convertSmallDataProxy called on non small Data Proxy");
    return true;
  }
  
  const Pds::TypeId& proxiedTypeId = getSmallDataProxiedType(xtc.get());
  
  if (cvt.isSplitType(proxiedTypeId)) {
    MsgLog(logger, error, "convertSmallDataProxy called on shared split type. Not implemented.");
    return true;
  }
  
  if (m_finalized) {
    MsgLog(logger, error, "Finalized already called");
    return true;
  }
  return false;
}
  
void SmallDataProxy::addEventProxy(const boost::shared_ptr<Pds::Xtc>& origXtc,
                                   std::vector<const std::type_info *> typeInfoPtrs)
{
  boost::shared_ptr<Pds::Xtc> xtc = uncompressIfNeeded(origXtc);
  if (addProxyChecksFailed(xtc, m_cvt)) return;
  if (m_evtForProxies.expired()) {
    MsgLog(logger, error, "addEventProxy called for null event");
    return;
  }
  
  MsgLog(logger, DEBUGMSG, "addEventProxy called. First type: " 
         << PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtrs.at(0)));
  
  if (1 != xtc->contains.compressed_version()) {
    MsgLog(logger, error, "addEventProxy called on version that is not 1, version is " << xtc->contains.compressed_version());
    return;
  }
  Pds::Src src = xtc->src;
  Pds::Damage damage = xtc->damage;
  Pds::SmlData::ProxyV1 * smlDataProxy = static_cast<Pds::SmlData::ProxyV1 *>(static_cast<void *>(xtc->payload()));
  int64_t fileOffset = smlDataProxy->fileOffset();
  uint32_t extent = smlDataProxy->extent();
  Pds::TypeId typeId = smlDataProxy->type();

  ObjId objId = getNextObjId();
  m_ids[objId] = ObjData(objId, fileOffset, extent, typeId, src, damage);
  MsgLog(logger, DEBUGMSG, "  addEventProxy - assigned objId=" << objId);
  std::vector<const std::type_info *>::iterator curType;

  boost::shared_ptr<PSEvt::Event> evtForProxies = m_evtForProxies.lock();
  if (not evtForProxies) {
    MsgLog(logger, error, "addEventProxy - the Event is null.");
    return;
  }
  for (curType = typeInfoPtrs.begin(); curType != typeInfoPtrs.end(); ++curType) {
    PSEvt::EventKey curEventKey(*curType, src, "");
    boost::shared_ptr<LargeObjectProxy> largeObjProxy = \
      LargeObjectProxy::makeLargeObjectProxy(*curType,
                                             objId,
                                             this->shared_from_this());
    try {
      evtForProxies->proxyDict()->put(largeObjProxy, curEventKey);
    } catch (PSEvt::ExceptionDuplicateKey &) {
      MsgLog(logger, warning, "smallData duplicate Exception while adding " << curEventKey 
             << " to the user event for proxies - ignoring.");
    }
  }
}
  

void SmallDataProxy::addEnvProxy(const boost::shared_ptr<Pds::Xtc>& origXtc,
                                 std::vector<const std::type_info *> typeInfoPtrs)
{
  boost::shared_ptr<Pds::Xtc> xtc = uncompressIfNeeded(origXtc);
  if (addProxyChecksFailed(xtc, m_cvt)) return;
  
  MsgLog(logger, DEBUGMSG, "addEnvProxy called. First type: " 
         << PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtrs.at(0)));

  if (1 != xtc->contains.compressed_version()) {
    MsgLog(logger, error, "addEnvProxy called on version that is not 1, version is " << xtc->contains.compressed_version());
    return;
  }

  Pds::Src src = xtc->src;
  Pds::Damage damage = xtc->damage;

  Pds::SmlData::ProxyV1 * smlDataProxy = static_cast<Pds::SmlData::ProxyV1 *>(static_cast<void *>(xtc->payload()));
  int64_t fileOffset = smlDataProxy->fileOffset();
  uint32_t extent = smlDataProxy->extent();
  Pds::TypeId typeId = smlDataProxy->type();

  ObjId objId = getNextObjId();
  m_ids[objId] = ObjData(objId, fileOffset, extent, typeId, src, damage);
  MsgLog(logger, DEBUGMSG, "  addEnvProxy - assigned objId=" << objId);

  std::vector<const std::type_info *>::iterator curType;

  boost::shared_ptr<PSEnv::EnvObjectStore> configStoreForProxies = m_configStoreForProxies.lock();
  
  if (not configStoreForProxies) {
    MsgLog(logger, error, "could not lock configStore pointer in addEnvProxy");
    return;
  }

  for (curType = typeInfoPtrs.begin(); curType != typeInfoPtrs.end(); ++curType) {
    PSEvt::EventKey curEventKey(*curType, src, "");
    boost::shared_ptr<LargeObjectProxy> largeObjProxy = \
      LargeObjectProxy::makeLargeObjectProxy(*curType,
                                             objId,
                                             this->shared_from_this());
    configStoreForProxies->proxyDict()->put(largeObjProxy, curEventKey);
  }
}

void SmallDataProxy::finalize() 
{ 
  // The main point of this function is to optimize the reads we do from the large
  // xtc file datagram for the potentially numerous proxied types. For instance, if the file
  // system reads in 4MB chunks, and we are proxying 4 1MB xtc objects that are contiguous, we
  // would like to read all 4 with one disk access rather then wait for the user to fetch each one
  // and use 4 disk accesses. 
  
  // To do this we introduce the idea of groups - each obj id will belong to a group. When a proxy
  // is triggered for any object in the group, all the objects will be loaded. 

  // For now, the strategy to form groups will be a straighforward grouping of the objects in order
  // until we go over the optimal read bytes parameter. I think there are better performing
  // strategies the data.

  if (m_finalized) {
    MsgLog(logger, fatal, "finalized has already been called");
    return;
  }

  std::vector<ObjData *> sortedObjData;
  
  std::map<ObjId, ObjData>::iterator pos;
  for (pos = m_ids.begin(); pos != m_ids.end(); ++pos) {
    sortedObjData.push_back(&(pos->second));
  }
  std::sort(sortedObjData.begin(), sortedObjData.end(), LessObjDataByFileOffset());

  // the current group id, starting offset, extent and object ids (will initialize with first data)
  GroupId groupId = 0;
  int64_t totalExtent = 0;
  int64_t groupStart = -1;
  std::vector<ObjId> idsToPutInThisGroup;

  for (unsigned pos = 0; pos < sortedObjData.size(); ++pos) {
    ObjData &objData = *(sortedObjData.at(pos));
    ObjId objId = objData.objId;
    int64_t fileOffset = objData.fileOffset;
    uint32_t extent = objData.extent;
    if (groupStart == -1) {
      // first object, initialize first group values
      groupId++;
      groupStart = fileOffset;
      totalExtent = extent;
      idsToPutInThisGroup.push_back(objId);
      continue;
    }
    if (totalExtent > m_optimalReadBytes) {
      // set group
      m_groups[groupId] = GroupData(idsToPutInThisGroup);
      for (unsigned idx = 0; idx < idsToPutInThisGroup.size(); ++idx) {
        ObjId idThisGroup = idsToPutInThisGroup.at(idx);
        m_ids[idThisGroup].groupId = groupId;
      }
      idsToPutInThisGroup.clear();
      // start new group
      groupId++;
      groupStart = fileOffset;
      totalExtent = extent;
      idsToPutInThisGroup.push_back(objId);
      continue;
    } 
    // add to group
    idsToPutInThisGroup.push_back(objId);
    totalExtent = (fileOffset + extent) - groupStart;
  }

  // create last group if needed
  if (idsToPutInThisGroup.size()>0) {
    m_groups[groupId] = GroupData(idsToPutInThisGroup);
    for (unsigned idx = 0; idx < idsToPutInThisGroup.size(); ++idx) {
      ObjId idThisGroup = idsToPutInThisGroup.at(idx);
      m_ids[idThisGroup].groupId = groupId;
    }
  }
  m_finalized = true; 
  MsgLog(logger, TRACEMSG, "finalized event. " << m_ids.size() << " proxied objects placed into "  
         << m_groups.size() << " groups");
  MsgLog(logger, DEBUGMSG, dumpStr());
}  

std::string SmallDataProxy::dumpStr() {
  std::ostringstream msg;
  msg << "xtcfile=" << m_smallXtcFileName
      << " live=" << m_liveMode
      << " finalized=" << m_finalized
      << " readBytes=" << m_optimalReadBytes
      << " num(ids)=" << m_ids.size()
      << " num(groups)=" << m_groups.size();
  if (not m_finalized) {
    msg << " not finalized";
  } else { // is finalized
    msg << std::endl;
    // dump ids
    std::map<ObjId, ObjData>::iterator idPos;
    for (idPos = m_ids.begin(); idPos != m_ids.end(); ++idPos) {
      ObjData & objData = idPos->second;
      ObjId curId = idPos->first;
      if (curId != objData.objId) MsgLog(logger, fatal, "m_ids map key is not stored in objData for id=" << curId);
      msg << " id=" << idPos->first
          << " group=" << objData.groupId
          << " fileOffset=" << objData.fileOffset
          << " extent=" << objData.extent
          << " type=" << objData.typeId.id()
          << " ver=" << objData.typeId.version()
          << std::endl;
    }
    // dump groups
    std::map<GroupId, GroupData>::iterator groupPos;
    for (groupPos = m_groups.begin(); groupPos != m_groups.end(); ++groupPos) {
      GroupData &groupData = groupPos->second;
      msg << " group=" << groupPos->first << " loaded=" << groupData.loaded << " ids=";
      std::vector<ObjId> &groupIds = groupData.ids;
      for (unsigned idx = 0; idx < groupIds.size(); ++idx) msg << " " << groupIds.at(idx);
      msg << "  ";
    }
    if (m_groups.size()>0) msg << std::endl;
    std::list<PSEvt::EventKey>::iterator pos;
    // dump keys in evtForLargeData:
    if (not m_evtForLargeData.get()) {
      msg << "  evtForLargeData is null" << std::endl;
    } else {
      std::list<PSEvt::EventKey> largeEvtKeys = m_evtForLargeData->keys();
      if (largeEvtKeys.size()==0) {
        msg << "  evtForLargeData is empty" << std::endl;
      } else {
        msg << "  evtForLargeData contains: " << std::endl;
        for ( pos = largeEvtKeys.begin(); pos != largeEvtKeys.end(); ++pos) {
          msg << "    " << *pos << std::endl;
        }
      }
    } 
    // dump keys in evtForProxies
    boost::shared_ptr<PSEvt::Event> evtForProxies = m_evtForProxies.lock();
    if (not evtForProxies) {
      msg << "  evtForProxies is null" << std::endl;
    } else {
      std::list<PSEvt::EventKey> usrEvtKeys = evtForProxies->keys();
      if (usrEvtKeys.size()==0) {
        msg << "  evtForProxies is empty" << std::endl;
      } else {
        msg << "  user Event contains: " << std::endl;
        for ( pos = usrEvtKeys.begin(); pos != usrEvtKeys.end(); ++pos) {
          msg << "    " << *pos << std::endl;
        }
      }
    }
  }
  return msg.str();
}

boost::shared_ptr<int8_t> SmallDataProxy::readBlockFromLargeFile(int64_t startOffset, int64_t extent) {
  MsgLog(logger, DEBUGMSG, "readBlockFromLargeFile(startOffset=" << startOffset << ", extent=" << extent <<")");
  if (not m_triedToOpen) {
    MsgLog(logger, DEBUGMSG, "  trying to open file");
    m_triedToOpen = true;
    if (not m_liveMode) {
      m_largeFile = XtcInput::SharedFile(m_largeXtcFileName,0);
      MsgLog(logger, DEBUGMSG, "  not live mode. opened file. largeFile.fd()=" << m_largeFile.fd());
    } else {
      unsigned timeWaited = 0;
      while ((timeWaited < m_liveTimeOut) and (m_largeFile.fd() == -1)) {
        try {
          m_largeFile = XtcInput::SharedFile(m_largeXtcFileName, m_liveTimeOut);
        } catch (XtcInput::FileOpenException &) {
          const unsigned sleepTime = 1;
          sleep(sleepTime);
          timeWaited += sleepTime;
          continue;
        }
        break;
      }
      if (m_largeFile.fd() == -1) {
        // we fatally exit here, otherwise with each event we will go through the live timeout
        MsgLog(logger, fatal, "unable to open large xtc file in live mode. Waited " 
               << timeWaited << " seconds, large filename: " << m_largeXtcFileName);
      }
      MsgLog(logger, DEBUGMSG, "  live mode. opened file. largeFile.fd()=" << m_largeFile.fd());
    }
  } // finish first call - try to open large file

  if (m_largeFile.fd() == -1) {
    MsgLog(logger, error, "no open file to read from");
    return boost::shared_ptr<int8_t>();
  }

  off_t seekedTo = m_largeFile.seek(startOffset, SEEK_SET);
  if (seekedTo != startOffset) {
    MsgLog(logger, error, "failed to seek to pos " << startOffset << " in large file " 
           << m_largeXtcFileName << " got to " << seekedTo);
    return boost::shared_ptr<int8_t>();
  }
      
  int8_t * p = new int8_t[extent];
  boost::shared_ptr<int8_t> buffer(p);
  
  ssize_t amountRead = m_largeFile.read(static_cast<char *>(static_cast<void *>(buffer.get())), extent);
  if (amountRead != extent) {
    MsgLog(logger, error, "failed to read expected amount: " << extent << " only read " 
           << amountRead << " in large file " << m_largeXtcFileName);
    return boost::shared_ptr<int8_t>();
  }
  return buffer;
}

void SmallDataProxy::loadGroup(GroupId groupId) {
  GroupData &groupData = m_groups[groupId];
  unsigned numberIdsInGroup = groupData.ids.size();
  if (numberIdsInGroup==0) {
    MsgLog(logger, error, "loadGroup for group " << groupId << " has not objects in it");
    return;
  }
  ObjId firstId = groupData.ids.at(0);
  int64_t fileStartOffset = m_ids[firstId].fileOffset;
  ObjId lastId = groupData.ids.at(numberIdsInGroup-1);
  int64_t fileEndOffset = m_ids[lastId].fileOffset + m_ids[lastId].extent;
  if (fileEndOffset <= fileStartOffset) {
    MsgLog(logger, error, "loadGroup for group " << groupId << " end byte (" << fileEndOffset << ") <= start byte (" << fileStartOffset << ")");
    return;
  }
  int64_t blockExtent = fileEndOffset - fileStartOffset;
  boost::shared_ptr<int8_t> groupBlock = readBlockFromLargeFile(fileStartOffset, blockExtent);
  if (not groupBlock) {
    MsgLog(logger, error, "readBlockFromLargeFile return null");
    return;
  }
  std::vector<ObjId> groupIds = groupData.ids;
  boost::shared_ptr<PSEnv::EnvObjectStore> configStoreForProxies = m_configStoreForProxies.lock();
  if (not configStoreForProxies) {
    MsgLog(logger, fatal, "loadGroup - configStore pointer is null");
    return;
  }
  for (unsigned objIdx = 0; objIdx < numberIdsInGroup; ++objIdx) {
    ObjId & objId = groupIds.at(objIdx);
    ObjData & objData = m_ids[objId];
    int64_t fileOffset = objData.fileOffset;
    uint32_t extent = objData.extent;
    if (fileOffset - fileStartOffset + extent > blockExtent) {
      MsgLog(logger, error, "obj id " << objId << " at position " 
             << objIdx << " in id list for group " << groupId 
             << " has an extent that goes past that of the last obj in the group");
      continue;
    }
    int64_t posRelToBlockStart = fileOffset - fileStartOffset;
    boost::shared_ptr<Pds::Xtc> xtc(groupBlock, (Pds::Xtc*)(groupBlock.get() + posRelToBlockStart));
    // check xtc
    Pds::TypeId &typeId = objData.typeId;
    Pds::Src &src = objData.src;
    Pds::Damage &damage = objData.damage;
    if ((xtc->contains.id() != typeId.id()) or (xtc->contains.version() != typeId.version()) or 
        (xtc->src.log() != src.log()) or (xtc->src.phy() != src.phy()) or (xtc->damage.value() != damage.value())) {
      MsgLog(logger, error, "mismatch between proxy information in smd and xtc in large file." 
             << "largeFile=" << m_largeXtcFileName << " offset=" << fileOffset
             << " proxy typeid=" << typeId.id() << " ver=" << typeId.version()
             << " large xtc typeid=" << xtc->contains.id() << " ver=" << xtc->contains.version()
             << " proxy src=" << src << " xtc.src=" << xtc->src
             << " proxy dmg=" << damage.value() << " xtc.damage=" << xtc->damage.value()
             << " SKIPPING");
      continue;
    }
    xtcConvert(xtc, m_evtForLargeData.get(), *configStoreForProxies);
  }
}

boost::shared_ptr<void> SmallDataProxy::get(ObjId objId, const PSEvt::EventKey &eventKey) {
  if (m_ids.find(objId) == m_ids.end()) {
     MsgLog(logger, error, "objId=" << objId << " not known? Internal error");
    return boost::shared_ptr<void>();
  }
  ObjData & objData = m_ids[objId];
  GroupId groupId = objData.groupId;
  if (m_groups.find(groupId) == m_groups.end()) {
    MsgLog(logger, error, "groupId=" << groupId << " looked up for objId=" << objId << " not known? Internal error");
    return boost::shared_ptr<void>();
  }
  MsgLog(logger, DEBUGMSG, "get(objId=" << objId << " eventKey=" << eventKey << ") group=" << groupId << ")");
  GroupData &groupData = m_groups[groupId];
  if (not groupData.loaded) {
    MsgLog(logger, DEBUGMSG, "  get -- group not loaded, loading group.");
    loadGroup(groupId);
    groupData.loaded = true;
  }  
  boost::shared_ptr<void> ptrFromEvent = m_evtForLargeData->proxyDict()->get(eventKey.typeinfo(), PSEvt::Source(eventKey.src()), eventKey.key(), NULL);
  boost::shared_ptr<PSEnv::EnvObjectStore> configStoreForProxies = m_configStoreForProxies.lock();
  if (not configStoreForProxies) {
    MsgLog(logger, fatal, "get: configStoreForProxies is null");
  }
  boost::shared_ptr<void> ptrFromConfigStore = configStoreForProxies->proxyDict()->get(eventKey.typeinfo(), PSEvt::Source(eventKey.src()), eventKey.key(), NULL);
  if (ptrFromEvent and ptrFromConfigStore) {
    MsgLog(logger, error, "Unexpected: small data proxy has loaded same event key into both event and configstore? key=" 
           << eventKey << " returning event object.");
    return ptrFromEvent;
  }
  if (ptrFromEvent) return ptrFromEvent;
  if (ptrFromConfigStore) return ptrFromConfigStore;
  MsgLog(logger, error, "Unexpected: group " << groupId 
         << " for id " << objId << " is loaded, but nothing available for event key: " << eventKey
         << std::endl << dumpStr());
  return boost::shared_ptr<void>();
}

}; // namespace psddl_pds2psana
