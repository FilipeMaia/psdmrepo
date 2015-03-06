//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcConverter...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_pds2psana/XtcConverter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/Exceptions.h"
#include "psddl_pds2psana/PnccdFullFrameV1Proxy.h"
#include "psddl_pds2psana/bld.ddl.h"
#include "psddl_pds2psana/epics.ddl.h"
#include "psddl_pds2psana/pnccd.ddl.h"
#include "psddl_pds2psana/dispatch.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psddl_pds2psana;

namespace {
  
  const char logger[] = "XtcConverter";


  template<typename FinalType>
  void 
  storeEpicsObject(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EpicsStore& eStore, long epicsStoreTag)
  {
    typedef typename FinalType::XtcType XtcType;
    typedef typename FinalType::PsanaType PsanaType;
    
    // XTC object
    boost::shared_ptr<XtcType> xptr(xtc, (XtcType*)(xtc->payload()));
    
    // create and store psana object in epics store
    boost::shared_ptr<Psana::Epics::EpicsPvHeader> obj = boost::make_shared<FinalType>(xptr);
    eStore.store(obj, xtc->src, NULL, epicsStoreTag);
  }


  /*
   * Build new Xtc from sub-object of the given xtc. Returned Xtc will reuse old Xtc header
   * except for TypeId and payload size but will use new payload.
   */
  // "destructor" for cloned xtc
  void buf_dealloc(Pds::Xtc* xtc) {
    delete [] reinterpret_cast<char*>(xtc);
  }
  boost::shared_ptr<Pds::Xtc>
  makeXtc(const Pds::Xtc& xtc, Pds::TypeId typeId, const char* payload, size_t payloadSize)
  {
    // allocate space for new xtc
    char* buf = new char[sizeof xtc + payloadSize];

    // copy header, replace typeId and extent size
    Pds::Xtc* newxtc = new (buf) Pds::Xtc(typeId, xtc.src, xtc.damage);
    newxtc->extent = sizeof xtc + payloadSize;

    // copy payload
    std::copy(payload, payload + payloadSize, buf + sizeof xtc);

    return boost::shared_ptr<Pds::Xtc>(newxtc, buf_dealloc);
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psddl_pds2psana {

//----------------
// Constructors --
//----------------
XtcConverter::XtcConverter ()

{
}

//--------------
// Destructor --
//--------------
XtcConverter::~XtcConverter ()
{
}

const std::vector<XtcConverter::SharedSplitXtcs::SplitEntry> XtcConverter::SharedSplitXtcs::emptyList;

XtcConverter::SharedSplitXtcs::SharedSplitXtcs() {
  std::vector<SplitEntry> sharedIpimbVer0(3), sharedIpimbVer1(3), sharedPimVer1(3);

  sharedIpimbVer0.at(0) = std::make_pair(Pds::TypeId(Pds::TypeId::Id_IpimbData,1),  storeToEvent);
  sharedIpimbVer0.at(1) = std::make_pair(Pds::TypeId(Pds::TypeId::Id_IpimbConfig,1), storeToConfig);
  sharedIpimbVer0.at(2) = std::make_pair(Pds::TypeId(Pds::TypeId::Id_IpmFex, 1), storeToEvent);
  m_sharedSplitMap[Pds::TypeId(Pds::TypeId::Id_SharedIpimb,0)] = sharedIpimbVer0;

  sharedIpimbVer1.at(0) = std::make_pair(Pds::TypeId(Pds::TypeId::Id_IpimbData,2),  storeToEvent);
  sharedIpimbVer1.at(1) = std::make_pair(Pds::TypeId(Pds::TypeId::Id_IpimbConfig,2), storeToConfig);
  sharedIpimbVer1.at(2) = std::make_pair(Pds::TypeId(Pds::TypeId::Id_IpmFex, 1), storeToEvent);
  m_sharedSplitMap[Pds::TypeId(Pds::TypeId::Id_SharedIpimb,1)] = sharedIpimbVer1;

  sharedPimVer1.at(0) = std::make_pair(Pds::TypeId(Pds::TypeId::Id_TM6740Config, 2), storeToConfig);
  sharedPimVer1.at(1) = std::make_pair(Pds::TypeId(Pds::TypeId::Id_PimImageConfig, 1), storeToConfig);
  sharedPimVer1.at(2) = std::make_pair(Pds::TypeId(Pds::TypeId::Id_Frame, 1), storeToEvent);
  m_sharedSplitMap[Pds::TypeId(Pds::TypeId::Id_SharedPim,1)] = sharedIpimbVer1;
}

bool XtcConverter::SharedSplitXtcs::isSplitType(const Pds::TypeId &typeId) const {
  return (m_sharedSplitMap.find(typeId) != m_sharedSplitMap.end());
}

const std::vector<XtcConverter::SharedSplitXtcs::SplitEntry> &
XtcConverter::SharedSplitXtcs::splitTypes(const Pds::TypeId &typeId) const {
  std::map<Pds::TypeId, 
           std::vector<XtcConverter::SharedSplitXtcs::SplitEntry>,
           XtcConverter::LessTypeId>::const_iterator pos;
  pos = m_sharedSplitMap.find(typeId);
  if (pos == m_sharedSplitMap.end()) return emptyList;
  return pos->second;
}

std::vector<Pds::TypeId> XtcConverter::splitTypes(const Pds::TypeId &typeId) const {
  std::vector<Pds::TypeId> results;
  const std::vector<SharedSplitXtcs::SplitEntry> &splitEntries = m_sharedSplit.splitTypes(typeId);
  for (unsigned idx = 0; idx < splitEntries.size(); ++idx) {
    results.push_back(splitEntries.at(idx).first);
  }
  return results;
}

bool XtcConverter::SharedSplitXtcs::equal(const Pds::TypeId &sharedTypeId, const unsigned entry, 
                                          const Pds::TypeId &entryTypeId, 
                                          const StoreTo entryStoreTo) 
{
  const std::vector<SplitEntry> & splitEntriesForSharedType = splitTypes(sharedTypeId);
  if (splitEntriesForSharedType.size() <= entry) return false;
  const SplitEntry & splitEntry = splitEntriesForSharedType.at(entry);
  bool agree = (splitEntry.first.id() == entryTypeId.id() and 
                splitEntry.first.version() == entryTypeId.version() and 
                splitEntry.second == entryStoreTo);
  return agree;
}

/**
 *  @brief Convert one object and store it in the event.
 */
void 
XtcConverter::convert(const boost::shared_ptr<Pds::Xtc>& xtc, PSEvt::Event& evt, PSEnv::EnvObjectStore& cfgStore)
{
  // protect from zero-payload data
  if (xtc->sizeofPayload() <= 0) return;

  const Pds::TypeId& origTypeId = xtc->contains;

  if (not m_sharedSplit.isSplitType(origTypeId)) {
    // all real stuff is done here
    return xtcConvert(xtc, &evt, cfgStore);
  }

  /*
   * Special case for Shared BLD data. We split them into their individual
   * components and store them as regular objects instead of one large
   * composite object. Components include both configuration and event data
   * objects so we update config store here as well.
   *
   * The components that the types are split into are specified in the
   * SharedSplitXtcs instance.
   */
  MsgLog(logger, debug, "shared type: " << Pds::TypeId::name(origTypeId.id()) << " version: " 
                        << origTypeId.version() << " will be split into sub xtcs");
  if (origTypeId.id() == Pds::TypeId::Id_SharedIpimb and origTypeId.version() == 0) {

    const Pds::Bld::BldDataIpimbV0* bld = reinterpret_cast<const Pds::Bld::BldDataIpimbV0*>(xtc->payload());
    Pds::TypeId typeId;
    boost::shared_ptr<Pds::Xtc> newxtc;

    const Pds::Ipimb::DataV1& ipimbData = bld->ipimbData();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpimbData, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipimbData), ipimbData._sizeof());
    this->convert(newxtc, evt, cfgStore);
    if (not m_sharedSplit.equal(origTypeId,0,typeId,SharedSplitXtcs::storeToEvent)) MsgLog(logger,fatal,"splitEntry 0 is wrong");

    const Pds::Ipimb::ConfigV1& ipimbConfig = bld->ipimbConfig();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpimbConfig, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipimbConfig), ipimbConfig._sizeof());
    this->convertConfig(newxtc, cfgStore);
    if (not m_sharedSplit.equal(origTypeId,1,typeId,SharedSplitXtcs::storeToConfig)) MsgLog(logger,fatal,"splitEntry 1 is wrong");

    const Pds::Lusi::IpmFexV1& ipmFexData = bld->ipmFexData();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpmFex, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipmFexData), ipmFexData._sizeof());
    this->convert(newxtc, evt, cfgStore);
    if (not m_sharedSplit.equal(origTypeId,2,typeId,SharedSplitXtcs::storeToEvent)) MsgLog(logger,fatal,"splitEntry 2 is wrong");

    return;

  } else if (origTypeId.id() == Pds::TypeId::Id_SharedIpimb and origTypeId.version() == 1) {

    const Pds::Bld::BldDataIpimbV1* bld = reinterpret_cast<const Pds::Bld::BldDataIpimbV1*>(xtc->payload());
    Pds::TypeId typeId;
    boost::shared_ptr<Pds::Xtc> newxtc;

    const Pds::Ipimb::DataV2& ipimbData = bld->ipimbData();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpimbData, 2);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipimbData), ipimbData._sizeof());
    this->convert(newxtc, evt, cfgStore);
    if (not m_sharedSplit.equal(origTypeId,0,typeId,SharedSplitXtcs::storeToEvent)) MsgLog(logger,fatal,"splitEntry 0 is wrong");

    const Pds::Ipimb::ConfigV2& ipimbConfig = bld->ipimbConfig();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpimbConfig, 2);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipimbConfig), ipimbConfig._sizeof());
    this->convertConfig(newxtc, cfgStore);
    if (not m_sharedSplit.equal(origTypeId,1,typeId,SharedSplitXtcs::storeToConfig)) MsgLog(logger,fatal,"splitEntry 1 is wrong");

    const Pds::Lusi::IpmFexV1& ipmFexData = bld->ipmFexData();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpmFex, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipmFexData), ipmFexData._sizeof());
    this->convert(newxtc, evt, cfgStore);
    if (not m_sharedSplit.equal(origTypeId,2,typeId,SharedSplitXtcs::storeToEvent)) MsgLog(logger,fatal,"splitEntry 2 is wrong");

    return;

  } else if (origTypeId.id() == Pds::TypeId::Id_SharedPim and origTypeId.version() == 1) {

    const Pds::Bld::BldDataPimV1* bld = reinterpret_cast<const Pds::Bld::BldDataPimV1*>(xtc->payload());
    Pds::TypeId typeId;
    boost::shared_ptr<Pds::Xtc> newxtc;

    const Pds::Pulnix::TM6740ConfigV2& camConfig = bld->camConfig();
    typeId = Pds::TypeId(Pds::TypeId::Id_TM6740Config, 2);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&camConfig), camConfig._sizeof());
    this->convertConfig(newxtc, cfgStore);
    if (not m_sharedSplit.equal(origTypeId,0,typeId,SharedSplitXtcs::storeToConfig)) MsgLog(logger,fatal,"splitEntry 0 is wrong");

    const Pds::Lusi::PimImageConfigV1& pimConfig = bld->pimConfig();
    typeId = Pds::TypeId(Pds::TypeId::Id_PimImageConfig, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&pimConfig), pimConfig._sizeof());
    this->convertConfig(newxtc, cfgStore);
    if (not m_sharedSplit.equal(origTypeId,1,typeId,SharedSplitXtcs::storeToConfig)) MsgLog(logger,fatal,"splitEntry 1 is wrong");

    const Pds::Camera::FrameV1& frame = bld->frame();
    typeId = Pds::TypeId(Pds::TypeId::Id_Frame, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&frame), frame._sizeof());
    this->convert(newxtc, evt, cfgStore);
    if (not m_sharedSplit.equal(origTypeId,2,typeId,SharedSplitXtcs::storeToEvent)) MsgLog(logger,fatal,"splitEntry 2 is wrong");

    return;

  }
  MsgLog(logger,fatal,"unexpected type id to split: typeId=" << origTypeId.id() << " version=" << origTypeId.version());
}

std::vector<const std::type_info *> 
XtcConverter::getConvertTypeInfoPtrs(const Pds::TypeId & typeId) const {
  if (not isSplitType(typeId)) return getXtcConvertTypeInfoPtrs(typeId);

  std::vector<const std::type_info *> typeInfoPtrs;

  std::vector<Pds::TypeId> componentTypeIds = splitTypes(typeId);
  for (unsigned idx = 0; idx < componentTypeIds.size(); ++idx) {
    std::vector<const std::type_info *> componentTypeInfoPtrs = 
                       getXtcConvertTypeInfoPtrs(componentTypeIds[idx]);
    for (unsigned jdx = 0; jdx < componentTypeInfoPtrs.size(); ++jdx) {
      typeInfoPtrs.push_back(componentTypeInfoPtrs[jdx]);
    }
  }
  return typeInfoPtrs;
}

/**
 *  @brief Convert one object and store it in the config store.
 */
void 
XtcConverter::convertConfig(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EnvObjectStore& cfgStore)
{
  // protect from zero-payload data
  if (xtc->sizeofPayload() <= 0) return;

  // all real stuff is done here
  xtcConvert(xtc, 0, cfgStore);
}

/**
 *  @brief Convert one object and store it in the epics store.
 */
void 
XtcConverter::convertEpics(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EpicsStore& eStore, long epicsStoreTag)
{
  const Pds::TypeId& typeId = xtc->contains;
  if(typeId.id() == Pds::TypeId::Id_Epics) {

    const Pds::Epics::EpicsPvHeader* pvhdr = (const Pds::Epics::EpicsPvHeader*)(xtc->payload());
    switch(pvhdr->dbrType()) {
    
    case Pds::Epics::DBR_TIME_STRING:
      ::storeEpicsObject<Epics::EpicsPvTimeString>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_TIME_SHORT:
      ::storeEpicsObject<Epics::EpicsPvTimeShort>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_TIME_FLOAT:
      ::storeEpicsObject<Epics::EpicsPvTimeFloat>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_TIME_ENUM:
      ::storeEpicsObject<Epics::EpicsPvTimeEnum>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_TIME_CHAR:
      ::storeEpicsObject<Epics::EpicsPvTimeChar>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_TIME_LONG:
      ::storeEpicsObject<Epics::EpicsPvTimeLong>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_TIME_DOUBLE:
      ::storeEpicsObject<Epics::EpicsPvTimeDouble>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_CTRL_STRING:
      ::storeEpicsObject<Epics::EpicsPvCtrlString>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_CTRL_SHORT:
      ::storeEpicsObject<Epics::EpicsPvCtrlShort>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_CTRL_FLOAT:
      ::storeEpicsObject<Epics::EpicsPvCtrlFloat>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_CTRL_ENUM:
      ::storeEpicsObject<Epics::EpicsPvCtrlEnum>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_CTRL_CHAR:
      ::storeEpicsObject<Epics::EpicsPvCtrlChar>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_CTRL_LONG:
      ::storeEpicsObject<Epics::EpicsPvCtrlLong>(xtc, eStore, epicsStoreTag);
      break;
    case Pds::Epics::DBR_CTRL_DOUBLE:
      ::storeEpicsObject<Epics::EpicsPvCtrlDouble>(xtc, eStore, epicsStoreTag);
      break;
    default:
      break;
    }
  }
  
}

} // namespace psddl_pds2psana
