//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class HdfConverter...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_hdf2psana/HdfConverter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/BldInfo.hh"
#include "pdsdata/xtc/DetInfo.hh"
#include "PSEvt/Exceptions.h"
#include "psddl_hdf2psana/HdfGroupName.h"
#include "psddl_hdf2psana/Exceptions.h"
#include "psddl_hdf2psana/ValueTypeProxy.h"
#include "psddl_hdf2psana/andor.ddl.h"
#include "psddl_hdf2psana/bld.ddlm.h"
#include "psddl_hdf2psana/camera.ddl.h"
#include "psddl_hdf2psana/encoder.ddl.h"
#include "psddl_hdf2psana/evr.ddlm.h"
#include "psddl_hdf2psana/fccd.ddl.h"
#include "psddl_hdf2psana/fli.ddl.h"
#include "psddl_hdf2psana/gsc16ai.ddl.h"
#include "psddl_hdf2psana/ipimb.ddl.h"
#include "psddl_hdf2psana/lusi.ddl.h"
#include "psddl_hdf2psana/oceanoptics.ddl.h"
#include "psddl_hdf2psana/princeton.ddl.h"
#include "psddl_hdf2psana/pulnix.ddl.h"
#include "psddl_hdf2psana/timepix.ddl.h"
#include "psddl_hdf2psana/usdusb.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const std::string logger = "HdfConverter";

  // name of the attribute holding schema version
  const std::string versionAttrName("_psddlSchemaVersion");

  // name of the attributes holding TypeId info
  const std::string typeIdTypeAttrName("_xtcTypeId.type");
  const std::string typeIdVersionAttrName("_xtcTypeId.version");

  // name of the attributes holding Src info
  const std::string srcAttrName("_xtcSrc");

  // name of the group holding EPICS data
  const std::string epicsGroupName("Epics::EpicsPv");

  // helper class to build Src from stored 64-bit code
  class _SrcBuilder : public Pds::Src {
  public:
    _SrcBuilder(uint64_t value) {
      _phy = uint32_t(value >> 32);
      _log = uint32_t(value);
    }
  };


  template<typename Type>
  void
  storeValueType(PSEvt::Event& evt, const hdf5pp::Group& group, uint64_t idx, const Pds::Src& src)
  {
    typedef typename Type::PsanaType PsanaType;
    typedef psddl_hdf2psana::ValueTypeProxy<Type> proxy_type;

    // store data
    boost::shared_ptr<PSEvt::Proxy<PsanaType> > proxy(boost::make_shared<proxy_type>(group, idx));
    evt.putProxy(proxy, src);
  }

  template<typename Type>
  void
  storeConfigType(PSEnv::EnvObjectStore& cfgStore, const hdf5pp::Group& group, const Pds::Src& src)
  {
    typedef typename Type::PsanaType PsanaType;

    // store data
    boost::shared_ptr<PsanaType> obj(boost::make_shared<Type>(group));
    cfgStore.put(obj, src);
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psddl_hdf2psana {

//----------------
// Constructors --
//----------------
HdfConverter::HdfConverter ()
{
}

//--------------
// Destructor --
//--------------
HdfConverter::~HdfConverter ()
{
}

/**
 *  @brief Convert one object and store it in the event.
 */
void
HdfConverter::convert(const hdf5pp::Group& group, uint64_t idx, PSEvt::Event& evt, PSEnv::EnvObjectStore& cfgStore)
try {

  const Pds::TypeId& typeId = this->typeId(group);
  const Pds::Src& src = this->source(group);

  int schema = schemaVersion(group);

  int version = typeId.version();
  switch(typeId.id()) {
  case Pds::TypeId::Any:
  case Pds::TypeId::Id_Xtc:
    break;
  case Pds::TypeId::Id_Frame:
    if (version == 1) evt.putProxy(Camera::make_FrameV1(schema, group, idx), src);
    break;
  case Pds::TypeId::Id_AcqWaveform:
    break;
  case Pds::TypeId::Id_AcqConfig:
    break;
  case Pds::TypeId::Id_TwoDGaussian:
    if (version == 1) evt.putProxy(Camera::make_TwoDGaussianV1(schema, group, idx), src);
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
    break;
  case Pds::TypeId::Id_pnCCDconfig:
    break;
  case Pds::TypeId::Id_Epics:
    break;
  case Pds::TypeId::Id_FEEGasDetEnergy:
    break;
  case Pds::TypeId::Id_EBeam:
    if (version == 0) ::storeValueType<Bld::BldDataEBeamV0>(evt, group, idx, src);
    if (version == 1) ::storeValueType<Bld::BldDataEBeamV1>(evt, group, idx, src);
    break;
  case Pds::TypeId::Id_PhaseCavity:
    break;
  case Pds::TypeId::Id_PrincetonFrame:
    if (version == 1) {
      if (boost::shared_ptr<Psana::Princeton::ConfigV1> cfgPtr = cfgStore.get(src)) {
        evt.putProxy(Princeton::make_FrameV1(schema, group, idx, cfgPtr), src);
        break;
      }
      if (boost::shared_ptr<Psana::Princeton::ConfigV2> cfgPtr = cfgStore.get(src)) {
        evt.putProxy(Princeton::make_FrameV1(schema, group, idx, cfgPtr), src);
        break;
      }
      if (boost::shared_ptr<Psana::Princeton::ConfigV3> cfgPtr = cfgStore.get(src)) {
        evt.putProxy(Princeton::make_FrameV1(schema, group, idx, cfgPtr), src);
        break;
      }
    }
    break;
  case Pds::TypeId::Id_PrincetonConfig:
    break;
  case Pds::TypeId::Id_EvrData:
    break;
  case Pds::TypeId::Id_FrameFccdConfig:
    break;
  case Pds::TypeId::Id_FccdConfig:
    break;
  case Pds::TypeId::Id_IpimbData:
    if (version == 1) evt.putProxy(Ipimb::make_DataV1(schema, group, idx), src);
    if (version == 2) evt.putProxy(Ipimb::make_DataV2(schema, group, idx), src);
    break;
  case Pds::TypeId::Id_IpimbConfig:
    break;
  case Pds::TypeId::Id_EncoderData:
    break;
  case Pds::TypeId::Id_EncoderConfig:
    break;
  case Pds::TypeId::Id_EvrIOConfig:
    break;
  case Pds::TypeId::Id_PrincetonInfo:
    break;
  case Pds::TypeId::Id_CspadElement:
    break;
  case Pds::TypeId::Id_CspadConfig:
    break;
  case Pds::TypeId::Id_IpmFexConfig:
    break;
  case Pds::TypeId::Id_IpmFex:
    evt.putProxy(Lusi::make_IpmFexV1(schema, group, idx), src);
    break;
  case Pds::TypeId::Id_DiodeFexConfig:
    break;
  case Pds::TypeId::Id_DiodeFex:
    evt.putProxy(Lusi::make_DiodeFexV1(schema, group, idx), src);
    break;
  case Pds::TypeId::Id_PimImageConfig:
    break;
  case Pds::TypeId::Id_SharedIpimb:
    break;
  case Pds::TypeId::Id_AcqTdcConfig:
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
    break;
  case Pds::TypeId::Id_FexampConfig:
    break;
  case Pds::TypeId::Id_FexampElement:
    break;
  case Pds::TypeId::Id_Gsc16aiConfig:
    break;
  case Pds::TypeId::Id_Gsc16aiData:
    if (version == 1) {
      if (boost::shared_ptr<Psana::Gsc16ai::ConfigV1> cfgPtr = cfgStore.get(src)) {
        evt.putProxy(Gsc16ai::make_DataV1(schema, group, idx, cfgPtr), src);
        break;
      }
    }
    break;
  case Pds::TypeId::Id_PhasicsConfig:
    break;
  case Pds::TypeId::Id_TimepixConfig:
    break;
  case Pds::TypeId::Id_TimepixData:
    if (version == 1) evt.putProxy(Timepix::make_DataV1(schema, group, idx), src);
    if (version == 2) evt.putProxy(Timepix::make_DataV2(schema, group, idx), src);
    break;
  case Pds::TypeId::Id_CspadCompressedElement:
    break;
  case Pds::TypeId::Id_OceanOpticsConfig:
    break;
  case Pds::TypeId::Id_OceanOpticsData:
    if (version == 1) {
      if (boost::shared_ptr<Psana::OceanOptics::ConfigV1> cfgPtr = cfgStore.get(src)) {
        evt.putProxy(OceanOptics::make_DataV1(schema, group, idx, cfgPtr), src);
        break;
      }
    }
    break;
  case Pds::TypeId::Id_EpicsConfig:
    break;
  case Pds::TypeId::Id_FliConfig:
    break;
  case Pds::TypeId::Id_FliFrame:
    if (version == 1) {
      if (boost::shared_ptr<Psana::Fli::ConfigV1> cfgPtr = cfgStore.get(src)) {
        evt.putProxy(Fli::make_FrameV1(schema, group, idx, cfgPtr), src);
        break;
      }
    }
    break;
  case Pds::TypeId::Id_QuartzConfig:
    break;
  case Pds::TypeId::Reserved1:
    break;
  case Pds::TypeId::Reserved2:
    break;
  case Pds::TypeId::Id_AndorConfig:
    break;
  case Pds::TypeId::Id_AndorFrame:
    if (version == 1) {
      if (boost::shared_ptr<Psana::Andor::ConfigV1> cfgPtr = cfgStore.get(src)) {
        evt.putProxy(Andor::make_FrameV1(schema, group, idx, cfgPtr), src);
        break;
      }
    }
    break;
  case Pds::TypeId::Id_UsdUsbData:
    if (version == 1) evt.putProxy(UsdUsb::make_DataV1(schema, group, idx), src);
    break;
  case Pds::TypeId::Id_UsdUsbConfig:
    break;
  case Pds::TypeId::Id_GMD:
    break;
  case Pds::TypeId::Id_SharedAcqADC:
    break;
  case Pds::TypeId::Id_OrcaConfig:
    break;
  case Pds::TypeId::NumberOf:
    break;
  }
} catch (const PSEvt::ExceptionDuplicateKey& ex) {
  MsgLog(logger, warning, ex.what());
}

/**
 *  @brief Convert one object and store it in the config store.
 */
void
HdfConverter::convertConfig(const hdf5pp::Group& group, uint64_t idx, PSEnv::EnvObjectStore& cfgStore)
try {

  const Pds::TypeId& typeId = this->typeId(group);
  const Pds::Src& src = this->source(group);

  int schema = schemaVersion(group);

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
    break;
  case Pds::TypeId::Id_TwoDGaussian:
    break;
  case Pds::TypeId::Id_Opal1kConfig:
    break;
  case Pds::TypeId::Id_FrameFexConfig:
    if (version == 1) cfgStore.putProxy(Camera::make_FrameFexConfigV1(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_EvrConfig:
    if (version == 5) ::storeConfigType<EvrData::ConfigV5>(cfgStore, group, src);
    break;
  case Pds::TypeId::Id_TM6740Config:
    if (version == 1) cfgStore.putProxy(Pulnix::make_TM6740ConfigV1(schema, group, -1), src);
    if (version == 2) cfgStore.putProxy(Pulnix::make_TM6740ConfigV2(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_ControlConfig:
    break;
  case Pds::TypeId::Id_pnCCDframe:
    break;
  case Pds::TypeId::Id_pnCCDconfig:
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
    if (version == 1) cfgStore.putProxy(Princeton::make_ConfigV1(schema, group, -1), src);
    if (version == 2) cfgStore.putProxy(Princeton::make_ConfigV2(schema, group, -1), src);
    if (version == 3) cfgStore.putProxy(Princeton::make_ConfigV3(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_EvrData:
    break;
  case Pds::TypeId::Id_FrameFccdConfig:
    if (version == 1) cfgStore.putProxy(Camera::make_FrameFccdConfigV1(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_FccdConfig:
    if (version == 1) cfgStore.putProxy(FCCD::make_FccdConfigV1(schema, group, -1), src);
    if (version == 2) cfgStore.putProxy(FCCD::make_FccdConfigV2(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_IpimbData:
    break;
  case Pds::TypeId::Id_IpimbConfig:
    if (version == 1) cfgStore.putProxy(Ipimb::make_ConfigV1(schema, group, -1), src);
    if (version == 2) cfgStore.putProxy(Ipimb::make_ConfigV2(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_EncoderData:
    break;
  case Pds::TypeId::Id_EncoderConfig:
    if (version == 1) cfgStore.putProxy(Encoder::make_ConfigV1(schema, group, -1), src);
    if (version == 2) cfgStore.putProxy(Encoder::make_ConfigV2(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_EvrIOConfig:
    break;
  case Pds::TypeId::Id_PrincetonInfo:
    break;
  case Pds::TypeId::Id_CspadElement:
    break;
  case Pds::TypeId::Id_CspadConfig:
    break;
  case Pds::TypeId::Id_IpmFexConfig:
    if (version == 1) cfgStore.putProxy(Lusi::make_IpmFexConfigV1(schema, group, -1), src);
    if (version == 2) cfgStore.putProxy(Lusi::make_IpmFexConfigV2(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_IpmFex:
    break;
  case Pds::TypeId::Id_DiodeFexConfig:
    if (version == 1) cfgStore.putProxy(Lusi::make_DiodeFexConfigV1(schema, group, -1), src);
    if (version == 2) cfgStore.putProxy(Lusi::make_DiodeFexConfigV2(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_DiodeFex:
    break;
  case Pds::TypeId::Id_PimImageConfig:
    if (version == 1) cfgStore.putProxy(Lusi::make_PimImageConfigV1(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_SharedIpimb:
    break;
  case Pds::TypeId::Id_AcqTdcConfig:
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
    break;
  case Pds::TypeId::Id_FexampConfig:
    break;
  case Pds::TypeId::Id_FexampElement:
    break;
  case Pds::TypeId::Id_Gsc16aiConfig:
    if (version == 1) cfgStore.putProxy(Gsc16ai::make_ConfigV1(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_Gsc16aiData:
    break;
  case Pds::TypeId::Id_PhasicsConfig:
    break;
  case Pds::TypeId::Id_TimepixConfig:
    if (version == 1) cfgStore.putProxy(Timepix::make_ConfigV1(schema, group, -1), src);
    if (version == 2) cfgStore.putProxy(Timepix::make_ConfigV2(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_TimepixData:
    break;
  case Pds::TypeId::Id_CspadCompressedElement:
    break;
  case Pds::TypeId::Id_OceanOpticsConfig:
    if (version == 1) cfgStore.putProxy(OceanOptics::make_ConfigV1(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_OceanOpticsData:
    break;
  case Pds::TypeId::Id_EpicsConfig:
    break;
  case Pds::TypeId::Id_FliConfig:
    if (version == 1) cfgStore.putProxy(Fli::make_ConfigV1(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_FliFrame:
    break;
  case Pds::TypeId::Id_QuartzConfig:
    break;
  case Pds::TypeId::Reserved1:
    break;
  case Pds::TypeId::Reserved2:
    break;
  case Pds::TypeId::Id_AndorConfig:
    if (version == 1) cfgStore.putProxy(Andor::make_ConfigV1(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_AndorFrame:
    break;
  case Pds::TypeId::Id_UsdUsbData:
    break;
  case Pds::TypeId::Id_UsdUsbConfig:
    if (version == 1) cfgStore.putProxy(UsdUsb::make_ConfigV1(schema, group, -1), src);
    break;
  case Pds::TypeId::Id_GMD:
    break;
  case Pds::TypeId::Id_SharedAcqADC:
    break;
  case Pds::TypeId::Id_OrcaConfig:
    break;
  case Pds::TypeId::NumberOf:
    break;
  }
} catch (const PSEvt::ExceptionDuplicateKey& ex) {
  MsgLog(logger, warning, ex.what());
}

/**
 *  @brief Convert one object and store it in the epics store.
 */
void
HdfConverter::convertEpics(const hdf5pp::Group& group, uint64_t idx, PSEnv::EpicsStore& eStore)
{
  const Pds::TypeId& typeId = this->typeId(group);
  if(typeId.id() == Pds::TypeId::Id_Epics) {
  }
}

/**
 *  @brief This method should be called to reset cache whenever some groups are closed
 */
void
HdfConverter::resetCache()
{
  m_schemaVersionCache.clear();
  m_isEpicsCache.clear();
  m_typeIdCache.clear();
  m_sourceCache.clear();
}

bool
HdfConverter::isEpics(const hdf5pp::Group& group, int levels) const
{
  // check cache first
  std::map<hdf5pp::Group, bool>::const_iterator it = m_isEpicsCache.find(group);
  if (it !=  m_isEpicsCache.end()) return it->second;

  // look at group name
  bool res = group.basename() == ::epicsGroupName;
  if (not res and levels > 0) {
    // try its parent
    hdf5pp::Group parent = group.parent();
    if (parent.valid()) res = isEpics(parent, levels - 1);
  }

  // update cache
  m_isEpicsCache.insert(std::make_pair(group, res));

  return res;
}

int
HdfConverter::schemaVersion(const hdf5pp::Group& group, int levels) const
{
  // with default argument call myself with correct level depending on type of group
  if (levels < 0) return schemaVersion(group, isEpics(group) ? 2 : 1);

  // check cache first
  std::map<hdf5pp::Group, int>::const_iterator it = m_schemaVersionCache.find(group);
  if (it !=  m_schemaVersionCache.end()) return it->second;

  // look at attribute
  int version = 0;
  hdf5pp::Attribute<int> attr = group.openAttr<int>(::versionAttrName);
  if (attr.valid()) {
    version = attr.read();
  } else if (levels > 0) {
    // try parent group if attribute is not there
    hdf5pp::Group parent = group.parent();
    if (parent.valid()) version = schemaVersion(parent, levels - 1);
  }

  // update cache
  m_schemaVersionCache.insert(std::make_pair(group, version));

  return version;
}

// Get TypeId for the group or its parent (and its grand-parent for EPICS),
Pds::TypeId
HdfConverter::typeId(const hdf5pp::Group& group, int levels) const
{
  // with default argument call myself with correct level depending on type of group
  if (levels < 0) return typeId(group, isEpics(group) ? 2 : 1);

  // check cache first
  std::map<hdf5pp::Group, Pds::TypeId>::const_iterator it = m_typeIdCache.find(group);
  if (it !=  m_typeIdCache.end()) return it->second;

  // look at attribute
  Pds::TypeId typeId(Pds::TypeId::Any, 0xffff);
  hdf5pp::Attribute<unsigned> attrType = group.openAttr<unsigned>(::typeIdTypeAttrName);
  hdf5pp::Attribute<unsigned> attrVersion = group.openAttr<unsigned>(::typeIdVersionAttrName);
  if (attrType.valid() and attrVersion.valid()) {
    // build TypeId from attributes
    typeId = Pds::TypeId(Pds::TypeId::Type(attrType.read()), attrVersion.read());
  } else if (levels > 0) {
    // try parent group if attribute is not there
    hdf5pp::Group parent = group.parent();
    if (parent.valid()) typeId = this->typeId(parent, levels - 1);
  } else if (levels == 0) {
    // guess type id from group name for top-level type group
    typeId = HdfGroupName::nameToTypeId(group.basename());
  }

  // update cache
  m_typeIdCache.insert(std::make_pair(group, typeId));

  return typeId;
}


// Get Source for the group (or its parent for EPICS),
Pds::Src
HdfConverter::source(const hdf5pp::Group& group, int levels) const
{
  // with default argument call myself with correct level depending on type of group
  if (levels < 0) return source(group, isEpics(group) ? 1 : 0);

  // check cache first
  std::map<hdf5pp::Group, Pds::Src>::const_iterator it = m_sourceCache.find(group);
  if (it !=  m_sourceCache.end()) return it->second;

  // look at attribute
  Pds::Src src(Pds::Level::NumberOfLevels);
  hdf5pp::Attribute<uint64_t> attrSrc = group.openAttr<uint64_t>(::srcAttrName);
  if (attrSrc.valid()) {
    // build TypeId from attributes
    src = ::_SrcBuilder(attrSrc.read());
  } else if (levels > 0) {
    // try parent group if attribute is not there
    hdf5pp::Group parent = group.parent();
    if (parent.valid()) src = this->source(parent, levels - 1);
  } else if (levels == 0) {
    // guess type id from group name for top-level type group
    src = HdfGroupName::nameToSource(group.basename());

    // some corrections needed for incorrectly stored names
    if (src == Pds::DetInfo(0, Pds::DetInfo::NoDetector, 0, Pds::DetInfo::NoDevice, 0)) {
      src = Pds::BldInfo(0, Pds::BldInfo::EBeam);
    } else if (src == Pds::DetInfo(0, Pds::DetInfo::NoDetector, 0, Pds::DetInfo::NoDevice, 1)) {
      src = Pds::BldInfo(0, Pds::BldInfo::PhaseCavity);
    } else if (src == Pds::DetInfo(0, Pds::DetInfo::NoDetector, 0, Pds::DetInfo::NoDevice, 2)) {
      src = Pds::BldInfo(0, Pds::BldInfo::FEEGasDetEnergy);
    }
  }

  // update cache
  m_sourceCache.insert(std::make_pair(group, src));

  return src;
}


} // namespace psddl_hdf2psana
