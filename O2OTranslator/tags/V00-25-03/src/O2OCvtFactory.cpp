//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OCvtFactory...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OCvtFactory.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/AcqirisConfigV1.h"
#include "H5DataTypes/AcqirisTdcConfigV1.h"
#include "H5DataTypes/AndorConfigV1.h"
#include "H5DataTypes/AndorFrameV1.h"
#include "H5DataTypes/BldDataEBeamV0.h"
#include "H5DataTypes/BldDataEBeamV1.h"
#include "H5DataTypes/BldDataEBeamV2.h"
#include "H5DataTypes/BldDataEBeamV3.h"
#include "H5DataTypes/BldDataFEEGasDetEnergy.h"
#include "H5DataTypes/BldDataGMDV0.h"
#include "H5DataTypes/BldDataGMDV1.h"
#include "H5DataTypes/BldDataIpimbV0.h"
#include "H5DataTypes/BldDataIpimbV1.h"
#include "H5DataTypes/BldDataPhaseCavity.h"
#include "H5DataTypes/BldDataPimV1.h"
#include "H5DataTypes/CameraFrameFexConfigV1.h"
#include "H5DataTypes/CameraFrameV1.h"
#include "H5DataTypes/CameraTwoDGaussianV1.h"
#include "H5DataTypes/ControlDataConfigV1.h"
#include "H5DataTypes/ControlDataConfigV2.h"
#include "H5DataTypes/CsPad2x2ConfigV1.h"
#include "H5DataTypes/CsPad2x2ConfigV2.h"
#include "H5DataTypes/CsPadConfigV1.h"
#include "H5DataTypes/CsPadConfigV2.h"
#include "H5DataTypes/CsPadConfigV3.h"
#include "H5DataTypes/CsPadConfigV4.h"
#include "H5DataTypes/EncoderConfigV1.h"
#include "H5DataTypes/EncoderConfigV2.h"
#include "H5DataTypes/EncoderDataV1.h"
#include "H5DataTypes/EncoderDataV2.h"
#include "H5DataTypes/EpicsPvHeader.h"
#include "H5DataTypes/EvrConfigV1.h"
#include "H5DataTypes/EvrConfigV2.h"
#include "H5DataTypes/EvrConfigV3.h"
#include "H5DataTypes/EvrConfigV4.h"
#include "H5DataTypes/EvrConfigV5.h"
#include "H5DataTypes/EvrConfigV6.h"
#include "H5DataTypes/EvrConfigV7.h"
#include "H5DataTypes/EvrDataV3.h"
#include "H5DataTypes/EvrIOConfigV1.h"
#include "H5DataTypes/FccdConfigV1.h"
#include "H5DataTypes/FccdConfigV2.h"
#include "H5DataTypes/FliConfigV1.h"
#include "H5DataTypes/FliFrameV1.h"
#include "H5DataTypes/Gsc16aiConfigV1.h"
#include "H5DataTypes/IpimbConfigV1.h"
#include "H5DataTypes/IpimbConfigV2.h"
#include "H5DataTypes/IpimbDataV1.h"
#include "H5DataTypes/IpimbDataV2.h"
#include "H5DataTypes/LusiDiodeFexConfigV1.h"
#include "H5DataTypes/LusiDiodeFexConfigV2.h"
#include "H5DataTypes/LusiDiodeFexV1.h"
#include "H5DataTypes/LusiIpmFexConfigV1.h"
#include "H5DataTypes/LusiIpmFexConfigV2.h"
#include "H5DataTypes/LusiIpmFexV1.h"
#include "H5DataTypes/LusiPimImageConfigV1.h"
#include "H5DataTypes/OceanOpticsConfigV1.h"
#include "H5DataTypes/Opal1kConfigV1.h"
#include "H5DataTypes/OrcaConfigV1.h"
#include "H5DataTypes/PnCCDConfigV1.h"
#include "H5DataTypes/PnCCDConfigV2.h"
#include "H5DataTypes/PrincetonConfigV1.h"
#include "H5DataTypes/PrincetonConfigV2.h"
#include "H5DataTypes/PrincetonConfigV3.h"
#include "H5DataTypes/PrincetonConfigV4.h"
#include "H5DataTypes/PrincetonConfigV5.h"
#include "H5DataTypes/PrincetonFrameV1.h"
#include "H5DataTypes/PrincetonFrameV2.h"
#include "H5DataTypes/PrincetonInfoV1.h"
#include "H5DataTypes/PulnixTM6740ConfigV1.h"
#include "H5DataTypes/PulnixTM6740ConfigV2.h"
#include "H5DataTypes/QuartzConfigV1.h"
#include "H5DataTypes/TimepixConfigV1.h"
#include "H5DataTypes/TimepixConfigV2.h"
#include "H5DataTypes/TimepixConfigV3.h"
#include "H5DataTypes/UsdUsbConfigV1.h"
#include "H5DataTypes/UsdUsbDataV1.h"
#include "O2OTranslator/AcqirisConfigV1Cvt.h"
#include "O2OTranslator/AcqirisDataDescV1Cvt.h"
#include "O2OTranslator/AcqirisTdcDataV1Cvt.h"
#include "O2OTranslator/CameraFrameV1Cvt.h"
#include "O2OTranslator/ConfigDataTypeCvt.h"
#include "O2OTranslator/CsPadElementV1Cvt.h"
#include "O2OTranslator/CsPadElementV2Cvt.h"
#include "O2OTranslator/CsPadCalibV1Cvt.h"
#include "O2OTranslator/CsPad2x2CalibV1Cvt.h"
#include "O2OTranslator/CsPad2x2ElementV1Cvt.h"
#include "O2OTranslator/EvtDataTypeCvtDef.h"
#include "O2OTranslator/EpicsDataTypeCvt.h"
#include "O2OTranslator/FliFrameV1Cvt.h"
#include "O2OTranslator/Gsc16aiDataV1Cvt.h"
#include "O2OTranslator/OceanOpticsDataV1Cvt.h"
#include "O2OTranslator/PnCCDFrameV1Cvt.h"
#include "O2OTranslator/PrincetonFrameCvt.h"
#include "O2OTranslator/TimepixDataV1Cvt.h"
#include "O2OTranslator/TimepixDataV2Cvt.h"
#include "pdsdata/xtc/TypeId.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using boost::make_shared;
using boost::shared_ptr;
using namespace H5DataTypes;

namespace {

  using O2OTranslator::O2OCvtFactory;

  template<typename ConfigType>
  void makeConfigCvt(O2OCvtFactory::DataTypeCvtList& cvts, const hdf5pp::Group& group,
      const std::string& typeGroupName, Pds::Src src, const O2OTranslator::CvtOptions& cvtOptions)
  {
    // For every config type we register two converters - regular config converter
    // which works for all sources except BLD, and default event data converter for
    // all BLD sources
    if (src.level() == Pds::Level::Reporter) {
      // case for BLD data
      cvts.push_back(make_shared<O2OTranslator::EvtDataTypeCvtDef<ConfigType> >(group, typeGroupName, src, cvtOptions, "config"));
    } else {
      cvts.push_back(make_shared<O2OTranslator::ConfigDataTypeCvt<ConfigType> >(group, typeGroupName, src));
    }
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

/**
 *  @brief Constructor instantiates converters for all known data types
 */
O2OCvtFactory::O2OCvtFactory(ConfigObjectStore& configStore, CalibObjectStore& calibStore,
    const O2OMetaData& metadata, const CvtOptions& cvtOptions)
  : m_configStore(configStore)
  , m_calibStore(calibStore)
  , m_metadata(metadata)
  , m_cvtOptions(cvtOptions)
  , m_groupCvtMap()
{
}

// Return the list of converters for given arguments.
O2OCvtFactory::DataTypeCvtList
O2OCvtFactory::getConverters(const hdf5pp::Group& group, Pds::TypeId typeId, Pds::Src src)
{
  // for epics only, all sources (which are usually different in PID only) are
  // processed by a single converter
  if (typeId.id() == Pds::TypeId::Id_Epics) {
    src = Pds::DetInfo(0, Pds::DetInfo::EpicsArch, 0, Pds::DetInfo::NoDevice, 0);
  }

  TypeSrcCvtMap& typeSrcCvtMap = m_groupCvtMap[group];

  TypeSrcCvtMap::iterator it = typeSrcCvtMap.find(std::make_pair(typeId, src));
  if (it != typeSrcCvtMap.end()) return it->second;

  // if not found make a full  list
  const DataTypeCvtList& cvts = makeCvts(group, typeId, src);
  typeSrcCvtMap.insert(TypeSrcCvtMap::value_type(std::make_pair(typeId, src), cvts));
  return cvts;
}

O2OCvtFactory::DataTypeCvtList
O2OCvtFactory::makeCvts(const hdf5pp::Group& group, Pds::TypeId typeId, Pds::Src src)
{
  DataTypeCvtList cvts;

  uint32_t version = typeId.version();
  switch(typeId.id()) {
  case Pds::TypeId::Any:
  case Pds::TypeId::Id_Xtc:
    break;

  case Pds::TypeId::Id_Frame:
    switch (version) {
    case 1:
      cvts.push_back(make_shared<CameraFrameV1Cvt>(group, "Camera::FrameV1", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_AcqWaveform:
    switch (version) {
    case 1:
      // very special converter for Acqiris::DataDescV1, it needs two types of data
      cvts.push_back(make_shared<AcqirisDataDescV1Cvt>(group, "Acqiris::DataDescV1", src, m_configStore, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_AcqConfig:
    // very special converters for Acqiris config data
    if (version == 1) {
      if (src.level() == Pds::Level::Reporter) {
        // case for BLD data
        cvts.push_back(make_shared<AcqirisConfigV1Cvt>(group, "Acqiris::ConfigV1", src, m_cvtOptions));
      } else {
        cvts.push_back(make_shared<O2OTranslator::ConfigDataTypeCvt<AcqirisConfigV1> >(group, "Acqiris::ConfigV1", src));
      }
    }
    break;

  case Pds::TypeId::Id_TwoDGaussian:
    switch (version) {
    case 1:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<CameraTwoDGaussianV1> >(group, "Camera::TwoDGaussianV1", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_Opal1kConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<Opal1kConfigV1>(cvts, group, "Opal1k::ConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_FrameFexConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<CameraFrameFexConfigV1>(cvts, group, "Camera::FrameFexConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_EvrConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<EvrConfigV1>(cvts, group, "EvrData::ConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<EvrConfigV2>(cvts, group, "EvrData::ConfigV2", src, m_cvtOptions);
      break;
    case 3:
      ::makeConfigCvt<EvrConfigV3>(cvts, group, "EvrData::ConfigV3", src, m_cvtOptions);
      break;
    case 4:
      ::makeConfigCvt<EvrConfigV4>(cvts, group, "EvrData::ConfigV4", src, m_cvtOptions);
      break;
    case 5:
      ::makeConfigCvt<EvrConfigV5>(cvts, group, "EvrData::ConfigV5", src, m_cvtOptions);
      break;
    case 6:
      ::makeConfigCvt<EvrConfigV6>(cvts, group, "EvrData::ConfigV6", src, m_cvtOptions);
      break;
    case 7:
      ::makeConfigCvt<EvrConfigV7>(cvts, group, "EvrData::ConfigV7", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_TM6740Config:
    switch (version) {
    case 1:
      ::makeConfigCvt<PulnixTM6740ConfigV1>(cvts, group, "Pulnix::TM6740ConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<PulnixTM6740ConfigV2>(cvts, group, "Pulnix::TM6740ConfigV2", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_ControlConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<ControlDataConfigV1>(cvts, group, "ControlData::ConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<ControlDataConfigV2>(cvts, group, "ControlData::ConfigV2", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_pnCCDframe:
    switch (version) {
    case 1:
      // very special converter for PNCCD::FrameV1, it needs two types of data
      cvts.push_back(make_shared<PnCCDFrameV1Cvt>(group, "PNCCD::FrameV1", src, m_configStore, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_pnCCDconfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<PnCCDConfigV1>(cvts, group, "PNCCD::ConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<PnCCDConfigV2>(cvts, group, "PNCCD::ConfigV2", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_Epics:
    switch (version) {
    case 1:
      // Epics converter, non-default chunk size
      cvts.push_back(make_shared<EpicsDataTypeCvt>(group, "Epics::EpicsPv", src, m_configStore, 16*1024, m_cvtOptions.compLevel()));
      break;
    }
    break;

  case Pds::TypeId::Id_FEEGasDetEnergy:
    switch (version) {
    case 0:
      // version for this type is 0
      cvts.push_back(make_shared<EvtDataTypeCvtDef<BldDataFEEGasDetEnergy> >(group, "Bld::BldDataFEEGasDetEnergy", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_EBeam:
    switch (version) {
    case 0:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<BldDataEBeamV0> >(group, "Bld::BldDataEBeamV0", src, m_cvtOptions));
      break;
    case 1:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<BldDataEBeamV1> >(group, "Bld::BldDataEBeamV1", src, m_cvtOptions));
      break;
    case 2:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<BldDataEBeamV2> >(group, "Bld::BldDataEBeamV2", src, m_cvtOptions));
      break;
    case 3:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<BldDataEBeamV3> >(group, "Bld::BldDataEBeamV3", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_PhaseCavity:
    switch (version) {
    case 0:
      // version for this type is 0
      cvts.push_back(make_shared<EvtDataTypeCvtDef<BldDataPhaseCavity> >(group, "Bld::BldDataPhaseCavity", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_PrincetonFrame:
    switch (version) {
    case 1:
      // very special converter for Princeton::FrameV1, it needs two types of data
      cvts.push_back(make_shared<PrincetonFrameCvt<PrincetonFrameV1>  >(group, "Princeton::FrameV1", src, m_configStore, m_cvtOptions));
      break;
    case 2:
      // very special converter for Princeton::FrameV2, it needs two types of data
      cvts.push_back(make_shared<PrincetonFrameCvt<PrincetonFrameV2>  >(group, "Princeton::FrameV2", src, m_configStore, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_PrincetonConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<PrincetonConfigV1>(cvts, group, "Princeton::ConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<PrincetonConfigV2>(cvts, group, "Princeton::ConfigV2", src, m_cvtOptions);
      break;
    case 3:
      ::makeConfigCvt<PrincetonConfigV3>(cvts, group, "Princeton::ConfigV3", src, m_cvtOptions);
      break;
    case 4:
      ::makeConfigCvt<PrincetonConfigV4>(cvts, group, "Princeton::ConfigV4", src, m_cvtOptions);
      break;
    case 5:
      ::makeConfigCvt<PrincetonConfigV4>(cvts, group, "Princeton::ConfigV5", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_EvrData:
    switch (version) {
    case 3:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<EvrDataV3> >(group, "EvrData::DataV3", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_FrameFccdConfig:
    // was never implemented in pdsdata
    break;

  case Pds::TypeId::Id_FccdConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<FccdConfigV1>(cvts, group, "FCCD::FccdConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<FccdConfigV2>(cvts, group, "FCCD::FccdConfigV2", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_IpimbData:
    switch (version) {
    case 1:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<H5DataTypes::IpimbDataV1> >(group, "Ipimb::DataV1", src, m_cvtOptions));
      break;
    case 2:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<H5DataTypes::IpimbDataV2> >(group, "Ipimb::DataV2", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_IpimbConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<H5DataTypes::IpimbConfigV1>(cvts, group, "Ipimb::ConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<H5DataTypes::IpimbConfigV2>(cvts, group, "Ipimb::ConfigV2", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_EncoderData:
    switch (version) {
    case 1:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<EncoderDataV1> >(group, "Encoder::DataV1", src, m_cvtOptions));
      break;
    case 2:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<EncoderDataV2> >(group, "Encoder::DataV2", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_EncoderConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<EncoderConfigV1>(cvts, group, "Encoder::ConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<EncoderConfigV2>(cvts, group, "Encoder::ConfigV2", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_EvrIOConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<EvrIOConfigV1>(cvts, group, "EvrData::IOConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_PrincetonInfo:
    switch (version) {
    case 1:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<PrincetonInfoV1> >(group, "Princeton::InfoV1", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_CspadElement:
    switch (version) {
    case 1:
      // very special converter for CsPad::ElementV1, it needs two types of data
      cvts.push_back(make_shared<CsPadElementV1Cvt>(group, "CsPad::ElementV1", src, m_configStore, m_calibStore, m_cvtOptions));
      break;
    case 2:
      // very special converter for CsPad::ElementV2, it needs two types of data
      cvts.push_back(make_shared<CsPadElementV2Cvt>(group, "CsPad::ElementV2", src, m_configStore, m_calibStore, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_CspadConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<CsPadConfigV1>(cvts, group, "CsPad::ConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<CsPadConfigV2>(cvts, group, "CsPad::ConfigV2", src, m_cvtOptions);
      break;
    case 3:
      ::makeConfigCvt<CsPadConfigV3>(cvts, group, "CsPad::ConfigV3", src, m_cvtOptions);
      break;
    case 4:
      ::makeConfigCvt<CsPadConfigV4>(cvts, group, "CsPad::ConfigV4", src, m_cvtOptions);
      break;
    }
    // special converter object for CsPad calibration data
    cvts.push_back(shared_ptr<CsPadCalibV1Cvt>(new CsPadCalibV1Cvt(group, "CsPad::CalibV1", src, m_metadata, m_calibStore)));
    if (version == 3) {
      // some cspad2x2 data was produced without Cspad2x2Config object but
      // with CspadConfig/3 instead
      cvts.push_back(shared_ptr<CsPad2x2CalibV1Cvt>(new CsPad2x2CalibV1Cvt(group, "CsPad2x2::CalibV1", src, m_metadata, m_calibStore)));
    }
    break;

  case Pds::TypeId::Id_IpmFexConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<LusiIpmFexConfigV1>(cvts, group, "Lusi::IpmFexConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<LusiIpmFexConfigV2>(cvts, group, "Lusi::IpmFexConfigV2", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_IpmFex:
    switch (version) {
    case 1:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<LusiIpmFexV1> >(group, "Lusi::IpmFexV1", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_DiodeFexConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<LusiDiodeFexConfigV1>(cvts, group, "Lusi::DiodeFexConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<LusiDiodeFexConfigV2>(cvts, group, "Lusi::DiodeFexConfigV2", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_DiodeFex:
    switch (version) {
    case 1:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<LusiDiodeFexV1> >(group, "Lusi::DiodeFexV1", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_PimImageConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<LusiPimImageConfigV1>(cvts, group, "Lusi::PimImageConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_SharedIpimb:
// ==== shared stuff is split now ===
//    switch (version) {
//    case 0:
//      cvts.push_back(make_shared<EvtDataTypeCvtDef<BldDataIpimbV0> >(group, "Bld::BldDataIpimbV0", src, m_cvtOptions));
//      break;
//    case 1:
//      cvts.push_back(make_shared<EvtDataTypeCvtDef<BldDataIpimbV1> >(group, "Bld::BldDataIpimbV1", src, m_cvtOptions));
//      break;
//    }
    break;

  case Pds::TypeId::Id_AcqTdcConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<AcqirisTdcConfigV1>(cvts, group, "Acqiris::AcqirisTdcConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_AcqTdcData:
    switch (version) {
    case 1:
      cvts.push_back(make_shared<AcqirisTdcDataV1Cvt>(group, "Acqiris::TdcDataV1", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_Index:
    // this is not meant to be converted
    break;

  case Pds::TypeId::Id_XampsConfig:
    // TODO: implement when pdsdata is ready
    break;

  case Pds::TypeId::Id_XampsElement:
    // TODO: implement when pdsdata is ready
    break;

  case Pds::TypeId::Id_Cspad2x2Element:
    switch (version) {
    case 1:
      // very special converter for CsPad2x2::ElementV1, it needs calibrations
      cvts.push_back(make_shared<CsPad2x2ElementV1Cvt>(group, "CsPad2x2::ElementV1", src, m_calibStore, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_SharedPim:
// ==== shared stuff is split now ===
//    switch (version) {
//    case 1:
//      cvts.push_back(make_shared<EvtDataTypeCvtDef<BldDataPimV1> >(group, "Bld::BldDataPimV1", src, m_cvtOptions));
//      break;
//    }
    break;

  case Pds::TypeId::Id_Cspad2x2Config:
    switch (version) {
    case 1:
      ::makeConfigCvt<CsPad2x2ConfigV1>(cvts, group, "CsPad2x2::ConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<CsPad2x2ConfigV2>(cvts, group, "CsPad2x2::ConfigV2", src, m_cvtOptions);
      break;
    }
    // special converter object for CsPad calibration data
    cvts.push_back(shared_ptr<CsPad2x2CalibV1Cvt>(new CsPad2x2CalibV1Cvt(group, "CsPad2x2::CalibV1", src, m_metadata, m_calibStore)));
    break;

  case Pds::TypeId::Id_FexampConfig:
    // TODO: implement when pdsdata is ready
    break;

  case Pds::TypeId::Id_FexampElement:
    // TODO: implement when pdsdata is ready
    break;

  case Pds::TypeId::Id_Gsc16aiConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<Gsc16aiConfigV1>(cvts, group, "Gsc16ai::ConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_Gsc16aiData:
    switch (version) {
    case 1:
      // very special converter for Gsc16ai::DataV1, it needs two types of data
      cvts.push_back(make_shared<Gsc16aiDataV1Cvt>(group, "Gsc16ai::DataV1", src, m_configStore, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_PhasicsConfig:
    // TODO: implement when pdsdata is ready
    break;

  case Pds::TypeId::Id_TimepixConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<TimepixConfigV1>(cvts, group, "Timepix::ConfigV1", src, m_cvtOptions);
      break;
    case 2:
      ::makeConfigCvt<TimepixConfigV2>(cvts, group, "Timepix::ConfigV2", src, m_cvtOptions);
      break;
    case 3:
      ::makeConfigCvt<TimepixConfigV3>(cvts, group, "Timepix::ConfigV3", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_TimepixData:
    switch (version) {
    case 1:
      // very special converter for Timepix::DataV1
      // Note that it makes group DataV2 as internally it converts DataV1 into DataV2
      cvts.push_back(make_shared<TimepixDataV1Cvt>(group, "Timepix::DataV2", src, m_cvtOptions));
      break;
    case 2:
      // very special converter for Timepix::DataV1
      cvts.push_back(make_shared<TimepixDataV2Cvt>(group, "Timepix::DataV2", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_CspadCompressedElement:
    break;

  case Pds::TypeId::Id_OceanOpticsConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<OceanOpticsConfigV1>(cvts, group, "OceanOptics::ConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_OceanOpticsData:
    switch (version) {
    case 1:
      // very special converter for OceanOptics::DataV1, it needs two types of data
      cvts.push_back(make_shared<OceanOpticsDataV1Cvt>(group, "OceanOptics::DataV1", src, m_configStore, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_EpicsConfig:
    // this is handled internally by regular epics converter
    break;

  case Pds::TypeId::Id_FliConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<FliConfigV1>(cvts, group, "Fli::ConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_FliFrame:
    switch (version) {
    case 1:
      // very special converter for Fli::FrameV1, it needs two types of data
      cvts.push_back(make_shared<FliFrameV1Cvt<FliFrameV1> >(group, "Fli::FrameV1", src, m_configStore, Pds::TypeId(Pds::TypeId::Id_FliConfig, 1), m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_QuartzConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<QuartzConfigV1>(cvts, group, "Quartz::ConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Reserved1:
    break;

  case Pds::TypeId::Reserved2:
    break;

  case Pds::TypeId::Id_AndorConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<AndorConfigV1>(cvts, group, "Andor::ConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_AndorFrame:
    switch (version) {
    case 1:
      // very special converter for Andor::FrameV1, it needs two types of data
      cvts.push_back(make_shared<FliFrameV1Cvt<AndorFrameV1> >(group, "Andor::FrameV1", src, m_configStore, Pds::TypeId(Pds::TypeId::Id_FliConfig, 1), m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_UsdUsbData:
    switch (version) {
    case 1:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<UsdUsbDataV1> >(group, "UsdUsb::DataV1", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_UsdUsbConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<UsdUsbConfigV1>(cvts, group, "UsdUsb::ConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::Id_GMD:
    switch (version) {
    case 0:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<BldDataGMDV0> >(group, "Bld::BldDataGMDV0", src, m_cvtOptions));
      break;
    case 1:
      cvts.push_back(make_shared<EvtDataTypeCvtDef<BldDataGMDV1> >(group, "Bld::BldDataGMDV1", src, m_cvtOptions));
      break;
    }
    break;

  case Pds::TypeId::Id_SharedAcqADC:
    // shared stuff is split now
    break;

  case Pds::TypeId::Id_OrcaConfig:
    switch (version) {
    case 1:
      ::makeConfigCvt<OrcaConfigV1>(cvts, group, "Orca::ConfigV1", src, m_cvtOptions);
      break;
    }
    break;

  case Pds::TypeId::NumberOf:
    break;
  }

  return cvts;
}

// Notify factory that the group is about to be closed.
void
O2OCvtFactory::closeGroup(const hdf5pp::Group& group)
{
  m_groupCvtMap.erase(group);
}

bool
O2OCvtFactory::TypeAndSourceCmp::operator()(const TypeAndSource& lhs, const TypeAndSource& rhs) const
{
  if (lhs.first.value() < rhs.first.value()) return true;
  if (lhs.first.value() > rhs.first.value()) return false;
  if (lhs.second.log() < rhs.second.log()) return true;
  if (lhs.second.log() > rhs.second.log()) return false;
  return lhs.second.phy() < rhs.second.phy();
};


} // namespace O2OTranslator
