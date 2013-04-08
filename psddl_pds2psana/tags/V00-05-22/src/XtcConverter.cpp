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
  storeEpicsObject(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EpicsStore& eStore)
  {
    typedef typename FinalType::XtcType XtcType;
    typedef typename FinalType::PsanaType PsanaType;
    
    // XTC object
    boost::shared_ptr<XtcType> xptr(xtc, (XtcType*)(xtc->payload()));
    
    // create and store psana object in epics store
    boost::shared_ptr<Psana::Epics::EpicsPvHeader> obj = boost::make_shared<FinalType>(xptr);
    eStore.store(obj, xtc->src);
  }


  /*
   * Build new Xtc from sub-object of the given xtc. Returned Xtc will reuse old Xtc heared
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

/**
 *  @brief Convert one object and store it in the event.
 */
void 
XtcConverter::convert(const boost::shared_ptr<Pds::Xtc>& xtc, PSEvt::Event& evt, PSEnv::EnvObjectStore& cfgStore)
{
  // protect from zero-payload data
  if (xtc->sizeofPayload() <= 0) return;

  const Pds::TypeId& typeId = xtc->contains;
  
  /*
   * Special case for Shared BLD data. We split them into their individual
   * components and store them as regular objects instead of one large
   * composite object. Components include both configuration and event data
   * objects so we update config store here as well.
   */
  if (typeId.id() == Pds::TypeId::Id_SharedIpimb and typeId.version() == 0) {

    const PsddlPds::Bld::BldDataIpimbV0* bld = reinterpret_cast<const PsddlPds::Bld::BldDataIpimbV0*>(xtc->payload());
    Pds::TypeId typeId;
    boost::shared_ptr<Pds::Xtc> newxtc;

    const PsddlPds::Ipimb::DataV1& ipimbData = bld->ipimbData();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpimbData, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipimbData), ipimbData._sizeof());
    this->convert(newxtc, evt, cfgStore);

    const PsddlPds::Ipimb::ConfigV1& ipimbConfig = bld->ipimbConfig();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpimbConfig, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipimbConfig), ipimbConfig._sizeof());
    this->convertConfig(newxtc, cfgStore);

    const PsddlPds::Lusi::IpmFexV1& ipmFexData = bld->ipmFexData();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpmFex, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipmFexData), ipmFexData._sizeof());
    this->convert(newxtc, evt, cfgStore);

    return;

  } else if (typeId.id() == Pds::TypeId::Id_SharedIpimb and typeId.version() == 1) {

    const PsddlPds::Bld::BldDataIpimbV1* bld = reinterpret_cast<const PsddlPds::Bld::BldDataIpimbV1*>(xtc->payload());
    Pds::TypeId typeId;
    boost::shared_ptr<Pds::Xtc> newxtc;

    const PsddlPds::Ipimb::DataV2& ipimbData = bld->ipimbData();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpimbData, 2);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipimbData), ipimbData._sizeof());
    this->convert(newxtc, evt, cfgStore);

    const PsddlPds::Ipimb::ConfigV2& ipimbConfig = bld->ipimbConfig();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpimbConfig, 2);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipimbConfig), ipimbConfig._sizeof());
    this->convertConfig(newxtc, cfgStore);

    const PsddlPds::Lusi::IpmFexV1& ipmFexData = bld->ipmFexData();
    typeId = Pds::TypeId(Pds::TypeId::Id_IpmFex, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&ipmFexData), ipmFexData._sizeof());
    this->convert(newxtc, evt, cfgStore);

    return;

  } else if (typeId.id() == Pds::TypeId::Id_SharedPim and typeId.version() == 1) {

    const PsddlPds::Bld::BldDataPimV1* bld = reinterpret_cast<const PsddlPds::Bld::BldDataPimV1*>(xtc->payload());
    Pds::TypeId typeId;
    boost::shared_ptr<Pds::Xtc> newxtc;

    const PsddlPds::Pulnix::TM6740ConfigV2& camConfig = bld->camConfig();
    typeId = Pds::TypeId(Pds::TypeId::Id_TM6740Config, 2);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&camConfig), camConfig._sizeof());
    this->convertConfig(newxtc, cfgStore);

    const PsddlPds::Lusi::PimImageConfigV1& pimConfig = bld->pimConfig();
    typeId = Pds::TypeId(Pds::TypeId::Id_PimImageConfig, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&pimConfig), pimConfig._sizeof());
    this->convertConfig(newxtc, cfgStore);

    const PsddlPds::Camera::FrameV1& frame = bld->frame();
    typeId = Pds::TypeId(Pds::TypeId::Id_Frame, 1);
    newxtc = ::makeXtc(*xtc, typeId, reinterpret_cast<const char*>(&frame), frame._sizeof());
    this->convert(newxtc, evt, cfgStore);

    return;

  }

  // add special proxy for full pnccd frame
  if (typeId.id() == Pds::TypeId::Id_pnCCDframe and typeId.version() == 1) {
    evt.putProxy<Psana::PNCCD::FullFrameV1>(boost::make_shared<PnccdFullFrameV1Proxy>(), xtc->src);
  }

  // all real stuff is done here
  xtcConvert(xtc, &evt, cfgStore);
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
XtcConverter::convertEpics(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EpicsStore& eStore)
{
  const Pds::TypeId& typeId = xtc->contains;
  if(typeId.id() == Pds::TypeId::Id_Epics) {

    const PsddlPds::Epics::EpicsPvHeader* pvhdr = (const PsddlPds::Epics::EpicsPvHeader*)(xtc->payload());
    switch(pvhdr->dbrType()) {
    
    case PsddlPds::Epics::DBR_TIME_STRING:
      ::storeEpicsObject<Epics::EpicsPvTimeString>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_TIME_SHORT:
      ::storeEpicsObject<Epics::EpicsPvTimeShort>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_TIME_FLOAT:
      ::storeEpicsObject<Epics::EpicsPvTimeFloat>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_TIME_ENUM:
      ::storeEpicsObject<Epics::EpicsPvTimeEnum>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_TIME_CHAR:
      ::storeEpicsObject<Epics::EpicsPvTimeChar>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_TIME_LONG:
      ::storeEpicsObject<Epics::EpicsPvTimeLong>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_TIME_DOUBLE:
      ::storeEpicsObject<Epics::EpicsPvTimeDouble>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_CTRL_STRING:
      ::storeEpicsObject<Epics::EpicsPvCtrlString>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_CTRL_SHORT:
      ::storeEpicsObject<Epics::EpicsPvCtrlShort>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_CTRL_FLOAT:
      ::storeEpicsObject<Epics::EpicsPvCtrlFloat>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_CTRL_ENUM:
      ::storeEpicsObject<Epics::EpicsPvCtrlEnum>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_CTRL_CHAR:
      ::storeEpicsObject<Epics::EpicsPvCtrlChar>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_CTRL_LONG:
      ::storeEpicsObject<Epics::EpicsPvCtrlLong>(xtc, eStore);
      break;
    case PsddlPds::Epics::DBR_CTRL_DOUBLE: 
      ::storeEpicsObject<Epics::EpicsPvCtrlDouble>(xtc, eStore);
      break;
    default:
      break;
    }
  }
  
}

} // namespace psddl_pds2psana