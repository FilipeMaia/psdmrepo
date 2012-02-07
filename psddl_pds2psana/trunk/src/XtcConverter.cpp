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
#include "psddl_pds2psana/EvtProxy.h"
#include "psddl_pds2psana/EvtProxyCfg.h"
#include "psddl_pds2psana/acqiris.ddl.h"
#include "psddl_pds2psana/bld.ddl.h"
#include "psddl_pds2psana/camera.ddl.h"
#include "psddl_pds2psana/control.ddl.h"
#include "psddl_pds2psana/cspad.ddl.h"
#include "psddl_pds2psana/cspad2x2.ddl.h"
#include "psddl_pds2psana/CsPadDataOrdered.h"
#include "psddl_pds2psana/encoder.ddl.h"
#include "psddl_pds2psana/epics.ddl.h"
#include "psddl_pds2psana/evr.ddl.h"
#include "psddl_pds2psana/fccd.ddl.h"
#include "psddl_pds2psana/gsc16ai.ddl.h"
#include "psddl_pds2psana/ipimb.ddl.h"
#include "psddl_pds2psana/lusi.ddl.h"
#include "psddl_pds2psana/opal1k.ddl.h"
#include "psddl_pds2psana/pnccd.ddl.h"
#include "psddl_pds2psana/princeton.ddl.h"
#include "psddl_pds2psana/pulnix.ddl.h"
#include "psddl_pds2psana/timepix.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psddl_pds2psana;

using psddl_pds2psana::Bld::pds_to_psana;
using psddl_pds2psana::Lusi::pds_to_psana;
using psddl_pds2psana::Princeton::pds_to_psana;

namespace {
  
  const char logger[] = "XtcConverter";

  template<typename FinalType>
  void 
  storeCfgObject(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EnvObjectStore& cfgStore)
  {
    typedef typename FinalType::XtcType XtcType;
    typedef typename FinalType::PsanaType PsanaType;
    
    // store XTC object in config store
    boost::shared_ptr<XtcType> xptr(xtc, (XtcType*)(xtc->payload()));
    cfgStore.put(xptr, xtc->src);
    
    // create and store psana object in config store
    boost::shared_ptr<PsanaType> obj(new FinalType(xptr));
    cfgStore.put(obj, xtc->src);
  }
  
  template<typename FinalType>
  void 
  storeDataProxy(const boost::shared_ptr<Pds::Xtc>& xtc, PSEvt::Event& evt)
  {
    typedef typename FinalType::XtcType XtcType;
    typedef typename FinalType::PsanaType PsanaType;
    
    // XTC data object
    boost::shared_ptr<XtcType> xptr(xtc, (XtcType*)(xtc->payload()));
    size_t xtcSize = xtc->sizeofPayload();

    // Proxy type
    typedef EvtProxy<PsanaType, FinalType, XtcType> ProxyType; 

    // store proxy
    try {
      evt.putProxy<PsanaType>(boost::make_shared<ProxyType>(xptr, xtcSize), xtc->src);
    } catch (const PSEvt::ExceptionDuplicateKey& ex) {
      MsgLog(logger, warning, ex.what());
    }
  }
  
  template<typename FinalType>
  void 
  storeDataProxyWithSize(const boost::shared_ptr<Pds::Xtc>& xtc, PSEvt::Event& evt)
  {
    typedef typename FinalType::XtcType XtcType;
    typedef typename FinalType::PsanaType PsanaType;
    
    // XTC data object
    boost::shared_ptr<XtcType> xptr(xtc, (XtcType*)(xtc->payload()));
    size_t xtcSize = xtc->sizeofPayload();
    
    // Proxy type
    typedef EvtProxy<PsanaType, FinalType, XtcType, true> ProxyType; 

    // store proxy
    try {
      evt.putProxy<PsanaType>(boost::make_shared<ProxyType>(xptr, xtcSize), xtc->src);
    } catch (const PSEvt::ExceptionDuplicateKey& ex) {
      MsgLog(logger, warning, ex.what());
    }
  }
  
  template<typename FinalType, typename XtcConfigType>
  bool 
  storeDataProxyCfg(const boost::shared_ptr<Pds::Xtc>& xtc, PSEvt::Event& evt, PSEnv::EnvObjectStore& cfgStore)
  {
    typedef typename FinalType::XtcType XtcType;
    typedef typename FinalType::PsanaType PsanaType;
    
    // get config object
    boost::shared_ptr<XtcConfigType> cfgPtr = cfgStore.get(xtc->src);
    if (not cfgPtr.get()) return false;
    
    // XTC data object
    boost::shared_ptr<XtcType> xptr(xtc, (XtcType*)(xtc->payload()));
    
    // Proxy type
    typedef EvtProxyCfg<PsanaType, FinalType, XtcType, XtcConfigType> ProxyType; 

    // store proxy
    try {
      evt.putProxy<PsanaType>(boost::make_shared<ProxyType>(xptr, cfgPtr), xtc->src);
    } catch (const PSEvt::ExceptionDuplicateKey& ex) {
      MsgLog(logger, warning, ex.what());
    }

    return true;
  }
  
  template<typename FinalType, typename XtcConfigType1, typename XtcConfigType2>
  bool 
  storeDataProxyCfg2(const boost::shared_ptr<Pds::Xtc>& xtc, PSEvt::Event& evt, PSEnv::EnvObjectStore& cfgStore)
  {
    if (storeDataProxyCfg<FinalType, XtcConfigType1>(xtc, evt, cfgStore)) return true;
    if (storeDataProxyCfg<FinalType, XtcConfigType2>(xtc, evt, cfgStore)) return true;
    return false;
  }
  
  template<typename PsanaType, typename XtcType>
  void 
  storeValueType(const boost::shared_ptr<Pds::Xtc>& xtc, PSEvt::Event& evt)
  {
    // XTC data object
    const XtcType& xdata = *(XtcType*)(xtc->payload());
    
    //convert XtcType to Psana type
    const PsanaType& data = pds_to_psana(xdata);
    
    // store data
    try {
      evt.put(boost::make_shared<PsanaType>(data), xtc->src);
    } catch (const PSEvt::ExceptionDuplicateKey& ex) {
      MsgLog(logger, warning, ex.what());
    }
  }
  
  template<typename PsanaType, typename XtcType>
  void 
  storeCfgValueType(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EnvObjectStore& cfgStore)
  {
    // XTC data object
    XtcType* xdata = (XtcType*)(xtc->payload());

    // store XTC object in config store
    boost::shared_ptr<XtcType> xptr(xtc, xdata);
    cfgStore.put(xptr, xtc->src);
    
    //convert XtcType to Psana type
    const PsanaType& data = pds_to_psana(*xdata);

    // create and store psana object in config store
    cfgStore.put(boost::make_shared<PsanaType>(data), xtc->src);
  }

  template<typename FinalType>
  void 
  storeEpicsObject(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EpicsStore& eStore)
  {
    typedef typename FinalType::XtcType XtcType;
    typedef typename FinalType::PsanaType PsanaType;
    
    // XTC object
    boost::shared_ptr<XtcType> xptr(xtc, (XtcType*)(xtc->payload()));
    
    // create and store psana object in epics store
    boost::shared_ptr<Psana::Epics::EpicsPvHeader> obj(new FinalType(xptr));
    eStore.store(obj, xtc->src);
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
  const Pds::TypeId& typeId = xtc->contains;
  //uint32_t size = xtc->sizeofPayload();
  
  int version = typeId.version();
  switch(typeId.id()) {
  case Pds::TypeId::Any:
  case Pds::TypeId::Id_Xtc:
    break;
  case Pds::TypeId::Id_Frame:
    if (version == 1) ::storeDataProxy<Camera::FrameV1>(xtc, evt);
    break;
  case Pds::TypeId::Id_AcqWaveform:
    if (version == 1) ::storeDataProxyCfg<Acqiris::DataDescV1, PsddlPds::Acqiris::ConfigV1>(xtc, evt, cfgStore);
    break;
  case Pds::TypeId::Id_AcqConfig:
    break;
  case Pds::TypeId::Id_TwoDGaussian:
    if (version == 1) ::storeDataProxy<Camera::TwoDGaussianV1>(xtc, evt);
    break;
  case Pds::TypeId::Id_Opal1kConfig:
    break;
  case Pds::TypeId::Id_FrameFexConfig:
    break;
  case Pds::TypeId::Id_EvrConfig:
    break;
  case Pds::TypeId::Id_TM6740Config:
    break;
  case Pds::TypeId::Id_ControlConfig:
    break;
  case Pds::TypeId::Id_pnCCDframe:
    if (version == 1) ::storeDataProxyCfg2<PNCCD::FrameV1, PsddlPds::PNCCD::ConfigV1, PsddlPds::PNCCD::ConfigV2>(xtc, evt, cfgStore);
    break;
  case Pds::TypeId::Id_pnCCDconfig:
    break;
  case Pds::TypeId::Id_Epics:
    break;
  case Pds::TypeId::Id_FEEGasDetEnergy:
    if (version == 0) ::storeValueType<Psana::Bld::BldDataFEEGasDetEnergy, PsddlPds::Bld::BldDataFEEGasDetEnergy>(xtc, evt);
    break;
  case Pds::TypeId::Id_EBeam:
    if (version == 0) ::storeValueType<Psana::Bld::BldDataEBeamV0, PsddlPds::Bld::BldDataEBeamV0>(xtc, evt);
    if (version == 1) ::storeValueType<Psana::Bld::BldDataEBeamV1, PsddlPds::Bld::BldDataEBeamV1>(xtc, evt);
    if (version == 2) ::storeValueType<Psana::Bld::BldDataEBeamV2, PsddlPds::Bld::BldDataEBeamV2>(xtc, evt);
    break;
  case Pds::TypeId::Id_PhaseCavity:
    if (version == 0) ::storeValueType<Psana::Bld::BldDataPhaseCavity, PsddlPds::Bld::BldDataPhaseCavity>(xtc, evt);
    break;
  case Pds::TypeId::Id_PrincetonFrame:
    if (version == 1) ::storeDataProxyCfg2<Princeton::FrameV1, PsddlPds::Princeton::ConfigV1, PsddlPds::Princeton::ConfigV2>(xtc, evt, cfgStore);
    break;
  case Pds::TypeId::Id_PrincetonConfig:
    break;
  case Pds::TypeId::Id_EvrData:
    if (version == 3) ::storeDataProxy<EvrData::DataV3>(xtc, evt);
    break;
  case Pds::TypeId::Id_FrameFccdConfig:
    break;
  case Pds::TypeId::Id_FccdConfig:
    break;
  case Pds::TypeId::Id_IpimbData:
    if (version == 1) ::storeDataProxy<Ipimb::DataV1>(xtc, evt);
    if (version == 2) ::storeDataProxy<Ipimb::DataV2>(xtc, evt);
    break;
  case Pds::TypeId::Id_IpimbConfig:
    break;
  case Pds::TypeId::Id_EncoderData:
    if (version == 1) ::storeDataProxy<Encoder::DataV1>(xtc, evt);
    if (version == 2) ::storeDataProxy<Encoder::DataV2>(xtc, evt);
    break;
  case Pds::TypeId::Id_EncoderConfig:
    break;
  case Pds::TypeId::Id_EvrIOConfig:
    break;
  case Pds::TypeId::Id_PrincetonInfo:
    if (version == 1) ::storeValueType<Psana::Princeton::InfoV1, PsddlPds::Princeton::InfoV1>(xtc, evt);
    break;
  case Pds::TypeId::Id_CspadElement:
    if (version == 1) ::storeDataProxyCfg2<CsPadDataOrdered<CsPad::DataV1, Psana::CsPad::ElementV1>, PsddlPds::CsPad::ConfigV1, PsddlPds::CsPad::ConfigV2>(xtc, evt, cfgStore);
    if (version == 2) ::storeDataProxyCfg2<CsPadDataOrdered<CsPad::DataV2, Psana::CsPad::ElementV2>, PsddlPds::CsPad::ConfigV2, PsddlPds::CsPad::ConfigV3>(xtc, evt, cfgStore);
    break;
  case Pds::TypeId::Id_CspadConfig:
    break;
  case Pds::TypeId::Id_IpmFexConfig:
    break;
  case Pds::TypeId::Id_IpmFex:
    if (version == 1) ::storeValueType<Psana::Lusi::IpmFexV1, PsddlPds::Lusi::IpmFexV1>(xtc, evt);
    break;
  case Pds::TypeId::Id_DiodeFexConfig:
    break;
  case Pds::TypeId::Id_DiodeFex:
    if (version == 1) ::storeValueType<Psana::Lusi::DiodeFexV1, PsddlPds::Lusi::DiodeFexV1>(xtc, evt);
    break;
  case Pds::TypeId::Id_PimImageConfig:
    break;
  case Pds::TypeId::Id_SharedIpimb:
    if (version == 0) ::storeDataProxy<Bld::BldDataIpimbV0>(xtc, evt);
    if (version == 1) ::storeDataProxy<Bld::BldDataIpimbV1>(xtc, evt);
    break;
  case Pds::TypeId::Id_AcqTdcConfig:
    break;
  case Pds::TypeId::Id_AcqTdcData:
    if (version == 1) ::storeDataProxyWithSize<Acqiris::TdcDataV1>(xtc, evt);
    break;
  case Pds::TypeId::Id_Index:
    break;
  case Pds::TypeId::Id_XampsConfig:
    break;
  case Pds::TypeId::Id_XampsElement:
    break;
  case Pds::TypeId::Id_Cspad2x2Element:
    if (version == 1) ::storeDataProxy<CsPad2x2::ElementV1>(xtc, evt);
    break;
  case Pds::TypeId::Id_SharedPim:
    if (version == 1) ::storeDataProxy<Bld::BldDataPimV1>(xtc, evt);
    break;
  case Pds::TypeId::Id_Cspad2x2Config:
    break;
  case Pds::TypeId::Id_FexampConfig:
    break;
  case Pds::TypeId::Id_FexampElement:
    break;
  case Pds::TypeId::Id_Gsc16aiConfig:
    break;
  case Pds::TypeId::Id_Gsc16aiData:
    if (version == 1) ::storeDataProxyCfg<Gsc16ai::DataV1, PsddlPds::Gsc16ai::ConfigV1>(xtc, evt, cfgStore);
    break;
  case Pds::TypeId::Id_PhasicsConfig:
    break;
  case Pds::TypeId::Id_TimepixConfig:
    break;
  case Pds::TypeId::Id_TimepixData:
    if (version == 1) ::storeDataProxy<Timepix::DataV1>(xtc, evt);
    break;
  case Pds::TypeId::NumberOf:
    break;

  }

}

/**
 *  @brief Convert one object and store it in the config store.
 */
void 
XtcConverter::convertConfig(const boost::shared_ptr<Pds::Xtc>& xtc, PSEnv::EnvObjectStore& cfgStore)
{
  const Pds::TypeId& typeId = xtc->contains;

  int version = typeId.version();
  switch(typeId.id()) {
  case Pds::TypeId::Any:
  case Pds::TypeId::Id_Xtc:
    break;
  case Pds::TypeId::Id_Frame:
    break;
  case Pds::TypeId::Id_AcqWaveform: 
    break;
  case Pds::TypeId::Id_AcqConfig:
    if (version == 1) ::storeCfgObject<Acqiris::ConfigV1>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_TwoDGaussian:
    break;
  case Pds::TypeId::Id_Opal1kConfig:
    if (version == 1) ::storeCfgObject<Opal1k::ConfigV1>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_FrameFexConfig:
    if (version == 1) ::storeCfgObject<Camera::FrameFexConfigV1>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_EvrConfig:
    if (version == 1) ::storeCfgObject<EvrData::ConfigV1>(xtc, cfgStore);
    if (version == 2) ::storeCfgObject<EvrData::ConfigV2>(xtc, cfgStore);
    if (version == 3) ::storeCfgObject<EvrData::ConfigV3>(xtc, cfgStore);
    if (version == 4) ::storeCfgObject<EvrData::ConfigV4>(xtc, cfgStore);
    if (version == 5) ::storeCfgObject<EvrData::ConfigV5>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_TM6740Config:
    if (version == 1) ::storeCfgObject<Pulnix::TM6740ConfigV1>(xtc, cfgStore);
    if (version == 2) ::storeCfgObject<Pulnix::TM6740ConfigV2>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_ControlConfig:
    if (version == 1) ::storeCfgObject<ControlData::ConfigV1>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_pnCCDframe:
    break;
  case Pds::TypeId::Id_pnCCDconfig:
    if (version == 1) ::storeCfgObject<PNCCD::ConfigV1>(xtc, cfgStore);
    if (version == 2) ::storeCfgObject<PNCCD::ConfigV2>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_Epics:
    break;
  case Pds::TypeId::Id_FEEGasDetEnergy:
    break;
  case Pds::TypeId::Id_EBeam:
    break;
  case Pds::TypeId::Id_PhaseCavity:
    break;
  case Pds::TypeId::Id_PrincetonFrame:
    break;
  case Pds::TypeId::Id_PrincetonConfig:
    if (version == 1) ::storeCfgObject<Princeton::ConfigV1>(xtc, cfgStore);
    if (version == 2) ::storeCfgObject<Princeton::ConfigV2>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_EvrData:
    break;
  case Pds::TypeId::Id_FrameFccdConfig:
    if (version == 1) ::storeCfgObject<Camera::FrameFccdConfigV1>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_FccdConfig:
    if (version == 1) ::storeCfgObject<FCCD::FccdConfigV1>(xtc, cfgStore);
    if (version == 2) ::storeCfgObject<FCCD::FccdConfigV2>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_IpimbData:
    break;
  case Pds::TypeId::Id_IpimbConfig:
    if (version == 1) ::storeCfgObject<Ipimb::ConfigV1>(xtc, cfgStore);
    if (version == 2) ::storeCfgObject<Ipimb::ConfigV2>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_EncoderData:
    break;
  case Pds::TypeId::Id_EncoderConfig:
    if (version == 1) ::storeCfgObject<Encoder::ConfigV1>(xtc, cfgStore);
    if (version == 2) ::storeCfgObject<Encoder::ConfigV2>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_EvrIOConfig:
    if (version == 1) ::storeCfgObject<EvrData::IOConfigV1>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_PrincetonInfo:
    break;
  case Pds::TypeId::Id_CspadElement:
    break;
  case Pds::TypeId::Id_CspadConfig:
    if (version == 1) ::storeCfgObject<CsPad::ConfigV1>(xtc, cfgStore);
    if (version == 2) ::storeCfgObject<CsPad::ConfigV2>(xtc, cfgStore);
    if (version == 3) ::storeCfgObject<CsPad::ConfigV3>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_IpmFexConfig:
    if (version == 1) ::storeCfgObject<Lusi::IpmFexConfigV1>(xtc, cfgStore);
    if (version == 2) ::storeCfgObject<Lusi::IpmFexConfigV2>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_IpmFex:
    break;
  case Pds::TypeId::Id_DiodeFexConfig:
    if (version == 1) ::storeCfgValueType<Psana::Lusi::DiodeFexConfigV1, PsddlPds::Lusi::DiodeFexConfigV1>(xtc, cfgStore);
    if (version == 2) ::storeCfgValueType<Psana::Lusi::DiodeFexConfigV2, PsddlPds::Lusi::DiodeFexConfigV2>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_DiodeFex:
    break;
  case Pds::TypeId::Id_PimImageConfig:
    if (version == 1) ::storeCfgValueType<Psana::Lusi::PimImageConfigV1, PsddlPds::Lusi::PimImageConfigV1>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_SharedIpimb:
    break;
  case Pds::TypeId::Id_AcqTdcConfig:
    if (version == 1) ::storeCfgObject<Acqiris::TdcConfigV1>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_AcqTdcData:
    break;
  case Pds::TypeId::Id_Index:
    break;
  case Pds::TypeId::Id_XampsConfig:
    break;
  case Pds::TypeId::Id_XampsElement:
    break;
  case Pds::TypeId::Id_Cspad2x2Element:
    break;
  case Pds::TypeId::Id_SharedPim:
    break;
  case Pds::TypeId::Id_Cspad2x2Config:
    if (version == 1) ::storeCfgObject<CsPad2x2::ConfigV1>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_FexampConfig:
    break;
  case Pds::TypeId::Id_FexampElement:
    break;
  case Pds::TypeId::Id_Gsc16aiConfig:
    if (version == 1) ::storeCfgObject<Gsc16ai::ConfigV1>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_Gsc16aiData:
    break;
  case Pds::TypeId::Id_PhasicsConfig:
    break;
  case Pds::TypeId::Id_TimepixConfig:
    if (version == 1) ::storeCfgObject<Timepix::ConfigV1>(xtc, cfgStore);
    break;
  case Pds::TypeId::Id_TimepixData:
    break;
  case Pds::TypeId::NumberOf:
    break;
  
  }

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
