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
#include "H5DataTypes/PnCCDConfigV1.h"
#include "H5DataTypes/PnCCDConfigV2.h"
#include "H5DataTypes/PrincetonConfigV1.h"
#include "H5DataTypes/PrincetonConfigV2.h"
#include "H5DataTypes/PrincetonConfigV3.h"
#include "H5DataTypes/PrincetonConfigV4.h"
#include "H5DataTypes/PrincetonInfoV1.h"
#include "H5DataTypes/PulnixTM6740ConfigV1.h"
#include "H5DataTypes/PulnixTM6740ConfigV2.h"
#include "H5DataTypes/QuartzConfigV1.h"
#include "H5DataTypes/TimepixConfigV1.h"
#include "H5DataTypes/TimepixConfigV2.h"
#include "H5DataTypes/UsdUsbConfigV1.h"
#include "H5DataTypes/UsdUsbDataV1.h"
#include "O2OTranslator/AcqirisDataDescV1Cvt.h"
#include "O2OTranslator/AcqirisTdcDataV1Cvt.h"
#include "O2OTranslator/CameraFrameV1Cvt.h"
#include "O2OTranslator/ConfigDataTypeCvt.h"
#include "O2OTranslator/CsPadElementV1Cvt.h"
#include "O2OTranslator/CsPadElementV2Cvt.h"
#include "O2OTranslator/CsPadCalibV1Cvt.h"
#include "O2OTranslator/CsPad2x2CalibV1Cvt.h"
#include "O2OTranslator/CsPad2x2ElementV1Cvt.h"
#include "O2OTranslator/EvrDataV3Cvt.h"
#include "O2OTranslator/EvtDataTypeCvtDef.h"
#include "O2OTranslator/EpicsDataTypeCvt.h"
#include "O2OTranslator/FliFrameV1Cvt.h"
#include "O2OTranslator/Gsc16aiDataV1Cvt.h"
#include "O2OTranslator/OceanOpticsDataV1Cvt.h"
#include "O2OTranslator/PnCCDFrameV1Cvt.h"
#include "O2OTranslator/PrincetonFrameV1Cvt.h"
#include "O2OTranslator/TimepixDataV1Cvt.h"
#include "O2OTranslator/TimepixDataV2Cvt.h"
#include "pdsdata/xtc/TypeId.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using boost::make_shared;

namespace {

  void registerCvt(O2OTranslator::O2OCvtFactory::CvtMap& cvtMap, Pds::TypeId::Type typeId, int version,
      const O2OTranslator::O2OCvtFactory::DataTypeCvtPtr& cvt)
  {
    uint32_t typeIdVal =  Pds::TypeId(typeId, version).value() ;
    cvtMap.insert(O2OTranslator::O2OCvtFactory::CvtMap::value_type(typeIdVal, cvt));
  }


  template<typename ConfigType>
  void registerConfigCvt(O2OTranslator::O2OCvtFactory::CvtMap& cvtMap, const std::string& typeGroupName,
      Pds::TypeId::Type typeId, int version)
  {
    ::registerCvt(cvtMap, typeId, version, make_shared<O2OTranslator::ConfigDataTypeCvt<ConfigType> >(typeGroupName));
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
    const O2OMetaData& metadata, int compression)
  : m_cvtMap()
{
  // instantiate all factories
  DataTypeCvtPtr converter ;


  //
  //  ========================== Converters for config types ===============================
  //

  ::registerConfigCvt<H5DataTypes::AcqirisConfigV1>(m_cvtMap, "Acqiris::ConfigV1", Pds::TypeId::Id_AcqConfig, 1);

  ::registerConfigCvt<H5DataTypes::AcqirisTdcConfigV1>(m_cvtMap, "Acqiris::AcqirisTdcConfigV1", Pds::TypeId::Id_AcqTdcConfig, 1);

  ::registerConfigCvt<H5DataTypes::Opal1kConfigV1>(m_cvtMap, "Opal1k::ConfigV1", Pds::TypeId::Id_Opal1kConfig, 1);

  ::registerConfigCvt<H5DataTypes::PulnixTM6740ConfigV1>(m_cvtMap, "Pulnix::TM6740ConfigV1", Pds::TypeId::Id_TM6740Config, 1);
  ::registerConfigCvt<H5DataTypes::PulnixTM6740ConfigV2>(m_cvtMap, "Pulnix::TM6740ConfigV2", Pds::TypeId::Id_TM6740Config, 2);

  ::registerConfigCvt<H5DataTypes::CameraFrameFexConfigV1>(m_cvtMap, "Camera::FrameFexConfigV1", Pds::TypeId::Id_FrameFexConfig, 1);

  ::registerConfigCvt<H5DataTypes::EvrConfigV1>(m_cvtMap, "EvrData::ConfigV1", Pds::TypeId::Id_EvrConfig, 1);
  ::registerConfigCvt<H5DataTypes::EvrConfigV2>(m_cvtMap, "EvrData::ConfigV2", Pds::TypeId::Id_EvrConfig, 2);
  ::registerConfigCvt<H5DataTypes::EvrConfigV3>(m_cvtMap, "EvrData::ConfigV3", Pds::TypeId::Id_EvrConfig, 3);
  ::registerConfigCvt<H5DataTypes::EvrConfigV4>(m_cvtMap, "EvrData::ConfigV4", Pds::TypeId::Id_EvrConfig, 4);
  ::registerConfigCvt<H5DataTypes::EvrConfigV5>(m_cvtMap, "EvrData::ConfigV5", Pds::TypeId::Id_EvrConfig, 5);
  ::registerConfigCvt<H5DataTypes::EvrConfigV6>(m_cvtMap, "EvrData::ConfigV6", Pds::TypeId::Id_EvrConfig, 6);
  ::registerConfigCvt<H5DataTypes::EvrConfigV7>(m_cvtMap, "EvrData::ConfigV7", Pds::TypeId::Id_EvrConfig, 7);

  ::registerConfigCvt<H5DataTypes::EvrIOConfigV1>(m_cvtMap, "EvrData::IOConfigV1", Pds::TypeId::Id_EvrIOConfig, 1);

  ::registerConfigCvt<H5DataTypes::ControlDataConfigV1>(m_cvtMap, "ControlData::ConfigV1", Pds::TypeId::Id_ControlConfig, 1);
  ::registerConfigCvt<H5DataTypes::ControlDataConfigV2>(m_cvtMap, "ControlData::ConfigV2", Pds::TypeId::Id_ControlConfig, 2);

  ::registerConfigCvt<H5DataTypes::PnCCDConfigV1>(m_cvtMap, "PNCCD::ConfigV1", Pds::TypeId::Id_pnCCDconfig, 1);
  ::registerConfigCvt<H5DataTypes::PnCCDConfigV2>(m_cvtMap, "PNCCD::ConfigV2", Pds::TypeId::Id_pnCCDconfig, 2);

  ::registerConfigCvt<H5DataTypes::PrincetonConfigV1>(m_cvtMap, "Princeton::ConfigV1", Pds::TypeId::Id_PrincetonConfig, 1);
  ::registerConfigCvt<H5DataTypes::PrincetonConfigV2>(m_cvtMap, "Princeton::ConfigV2", Pds::TypeId::Id_PrincetonConfig, 2);
  ::registerConfigCvt<H5DataTypes::PrincetonConfigV3>(m_cvtMap, "Princeton::ConfigV3", Pds::TypeId::Id_PrincetonConfig, 3);
  ::registerConfigCvt<H5DataTypes::PrincetonConfigV4>(m_cvtMap, "Princeton::ConfigV4", Pds::TypeId::Id_PrincetonConfig, 4);

  ::registerConfigCvt<H5DataTypes::FccdConfigV1>(m_cvtMap, "FCCD::FccdConfigV1", Pds::TypeId::Id_FccdConfig, 1);
  ::registerConfigCvt<H5DataTypes::FccdConfigV2>(m_cvtMap, "FCCD::FccdConfigV2", Pds::TypeId::Id_FccdConfig, 2);

  ::registerConfigCvt<H5DataTypes::IpimbConfigV1>(m_cvtMap, "Ipimb::ConfigV1", Pds::TypeId::Id_IpimbConfig, 1);
  ::registerConfigCvt<H5DataTypes::IpimbConfigV2>(m_cvtMap, "Ipimb::ConfigV2", Pds::TypeId::Id_IpimbConfig, 2);

  ::registerConfigCvt<H5DataTypes::EncoderConfigV1>(m_cvtMap, "Encoder::ConfigV1", Pds::TypeId::Id_EncoderConfig, 1);
  ::registerConfigCvt<H5DataTypes::EncoderConfigV2>(m_cvtMap, "Encoder::ConfigV2", Pds::TypeId::Id_EncoderConfig, 2);

  ::registerConfigCvt<H5DataTypes::LusiDiodeFexConfigV1>(m_cvtMap, "Lusi::DiodeFexConfigV1", Pds::TypeId::Id_DiodeFexConfig, 1);
  ::registerConfigCvt<H5DataTypes::LusiDiodeFexConfigV2>(m_cvtMap, "Lusi::DiodeFexConfigV2", Pds::TypeId::Id_DiodeFexConfig, 2);

  ::registerConfigCvt<H5DataTypes::LusiIpmFexConfigV1>(m_cvtMap, "Lusi::IpmFexConfigV1", Pds::TypeId::Id_IpmFexConfig, 1);
  ::registerConfigCvt<H5DataTypes::LusiIpmFexConfigV2>(m_cvtMap, "Lusi::IpmFexConfigV2", Pds::TypeId::Id_IpmFexConfig, 2);

  ::registerConfigCvt<H5DataTypes::LusiPimImageConfigV1>(m_cvtMap, "Lusi::PimImageConfigV1", Pds::TypeId::Id_PimImageConfig, 1);

  ::registerConfigCvt<H5DataTypes::CsPadConfigV1>(m_cvtMap, "CsPad::ConfigV1", Pds::TypeId::Id_CspadConfig, 1);
  ::registerConfigCvt<H5DataTypes::CsPadConfigV2>(m_cvtMap, "CsPad::ConfigV2", Pds::TypeId::Id_CspadConfig, 2);
  ::registerConfigCvt<H5DataTypes::CsPadConfigV3>(m_cvtMap, "CsPad::ConfigV3", Pds::TypeId::Id_CspadConfig, 3);
  ::registerConfigCvt<H5DataTypes::CsPadConfigV4>(m_cvtMap, "CsPad::ConfigV4", Pds::TypeId::Id_CspadConfig, 4);

  ::registerConfigCvt<H5DataTypes::Gsc16aiConfigV1>(m_cvtMap, "Gsc16ai::ConfigV1", Pds::TypeId::Id_Gsc16aiConfig, 1);

  ::registerConfigCvt<H5DataTypes::TimepixConfigV1>(m_cvtMap, "Timepix::ConfigV1", Pds::TypeId::Id_TimepixConfig, 1);
  ::registerConfigCvt<H5DataTypes::TimepixConfigV2>(m_cvtMap, "Timepix::ConfigV2", Pds::TypeId::Id_TimepixConfig, 2);

  ::registerConfigCvt<H5DataTypes::CsPad2x2ConfigV1>(m_cvtMap, "CsPad2x2::ConfigV1", Pds::TypeId::Id_Cspad2x2Config, 1);

  ::registerConfigCvt<H5DataTypes::OceanOpticsConfigV1>(m_cvtMap, "OceanOptics::ConfigV1", Pds::TypeId::Id_OceanOpticsConfig, 1);

  ::registerConfigCvt<H5DataTypes::FliConfigV1>(m_cvtMap, "Fli::ConfigV1", Pds::TypeId::Id_FliConfig, 1);

  ::registerConfigCvt<H5DataTypes::QuartzConfigV1>(m_cvtMap, "Quartz::ConfigV1", Pds::TypeId::Id_QuartzConfig, 1);

  ::registerConfigCvt<H5DataTypes::AndorConfigV1>(m_cvtMap, "Andor::ConfigV1", Pds::TypeId::Id_AndorConfig, 1);

  ::registerConfigCvt<H5DataTypes::UsdUsbConfigV1>(m_cvtMap, "UsdUsb::ConfigV1", Pds::TypeId::Id_UsdUsbConfig, 1);

  // special converter object for CsPad calibration data
  converter.reset(new CsPadCalibV1Cvt("CsPad::CalibV1", metadata, calibStore));
  for (int v = 0; v < 256; ++ v) {
    ::registerCvt(m_cvtMap, Pds::TypeId::Id_CspadConfig, v, converter);
  }

  // special converter object for CsPad calibration data
  converter.reset(new CsPad2x2CalibV1Cvt("CsPad2x2::CalibV1", metadata, calibStore));
  for (int v = 0; v < 256; ++ v) {
    ::registerCvt(m_cvtMap, Pds::TypeId::Id_Cspad2x2Config, v, converter);
  }
  // some cspad2x2 data was produced without Cspad2x2Config object but
  // with CspadConfig/3 instead
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_CspadConfig, 3, converter);


  //
  //    ===================================  Converters for regular data ======================================
  //


  hsize_t chunk_size = 16*1024*1024 ;

  // instantiate all factories for event converters
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::CameraTwoDGaussianV1> >(
      "Camera::TwoDGaussianV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_TwoDGaussian, 1, converter);

  // version for this type is 0
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::BldDataFEEGasDetEnergy> >(
      "Bld::BldDataFEEGasDetEnergy", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_FEEGasDetEnergy, 0, converter);

  // version for this type is 0
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::BldDataEBeamV0> >(
      "Bld::BldDataEBeamV0", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_EBeam, 0, converter);

  // version for this type is 1
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::BldDataEBeamV1> >(
      "Bld::BldDataEBeamV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_EBeam, 1, converter);

  // version for this type is 2
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::BldDataEBeamV2> >(
      "Bld::BldDataEBeamV2", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_EBeam, 2, converter);

  // version for this type is 3
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::BldDataEBeamV3> >(
      "Bld::BldDataEBeamV3", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_EBeam, 3, converter);

  // version for this type is 0
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::BldDataIpimbV0> >(
      "Bld::BldDataIpimbV0", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_SharedIpimb, 0, converter);

  // version for this type is 1
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::BldDataIpimbV1> >(
      "Bld::BldDataIpimbV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_SharedIpimb, 1, converter);

  // version for this type is 0
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::BldDataPhaseCavity> >(
      "Bld::BldDataPhaseCavity", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_PhaseCavity, 0, converter);

  // version for this type is 1
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::BldDataPimV1> >(
      "Bld::BldDataPimV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_SharedPim, 1, converter);

  // version for this type is 0
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::BldDataGMDV0> >(
      "Bld::BldDataGMDV0", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_GMD, 0, converter);

  // version for this type is 1
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::BldDataGMDV1> >(
      "Bld::BldDataGMDV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_GMD, 1, converter);

  // version for this type is 1
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::EncoderDataV1> >(
      "Encoder::DataV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_EncoderData, 1, converter);

  // version for this type is 2
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::EncoderDataV2> >(
      "Encoder::DataV2", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_EncoderData, 2, converter);

  // version for this type is 1
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::IpimbDataV1> >(
      "Ipimb::DataV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_IpimbData, 1, converter);

  // version for this type is 1
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::IpimbDataV2> >(
      "Ipimb::DataV2", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_IpimbData, 2, converter);

  // version for this type is 1
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::UsdUsbDataV1> >(
      "UsdUsb::DataV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_UsdUsbData, 1, converter);

  // version for this type is 3
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::EvrDataV3> >(
      "EvrData::DataV3", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_EvrData, 3, converter);

  // special converter for CameraFrame type
  converter = make_shared<CameraFrameV1Cvt>("Camera::FrameV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_Frame, 1, converter);

  // very special converter for Acqiris::DataDescV1, it needs two types of data
  converter = make_shared<AcqirisDataDescV1Cvt>("Acqiris::DataDescV1", configStore, chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_AcqWaveform, 1, converter);

  // very special converter for Acqiris::TdcDataV1
  converter = make_shared<AcqirisTdcDataV1Cvt>("Acqiris::TdcDataV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_AcqTdcData, 1, converter);

  // very special converter for PNCCD::FrameV1, it needs two types of data
  converter = make_shared<PnCCDFrameV1Cvt>("PNCCD::FrameV1", configStore, chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_pnCCDframe, 1, converter);

  // very special converter for Princeton::FrameV1, it needs two types of data
  converter = make_shared<PrincetonFrameV1Cvt>("Princeton::FrameV1", configStore, chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_PrincetonFrame, 1, converter);

  // version for this type is 1
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::PrincetonInfoV1> >(
      "Princeton::InfoV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_PrincetonInfo, 1, converter);

  // temporary/diagnostics  Epics converter>(headers only)
//  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::EpicsPvHeader> >(
//      "Epics::EpicsPvHeader", chunk_size, compression);
//  ::registerCvt(m_cvtMap, Pds::TypeId::Id_Epics, 1, converter);

  // Epics converter, non-default chunk size
  converter = make_shared<EpicsDataTypeCvt>("Epics::EpicsPv", configStore, 16*1024, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_Epics, 1, converter);

  // version for this type is 1
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::LusiDiodeFexV1> >(
      "Lusi::DiodeFexV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_DiodeFex, 1, converter);

  // version for this type is 1
  converter = make_shared<EvtDataTypeCvtDef<H5DataTypes::LusiIpmFexV1> >(
      "Lusi::IpmFexV1", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_IpmFex, 1, converter);

  // very special converter for CsPad::ElementV1, it needs two types of data
  converter = make_shared<CsPadElementV1Cvt>("CsPad::ElementV1", configStore,
                                           calibStore, chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_CspadElement, 1, converter);

  // very special converter for CsPad::ElementV2, it needs two types of data
  converter = make_shared<CsPadElementV2Cvt>("CsPad::ElementV2", configStore,
                                           calibStore, chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_CspadElement, 2, converter);

  // very special converter for CsPad2x2::ElementV1, it needs calibrations
  converter = make_shared<CsPad2x2ElementV1Cvt>(
      "CsPad2x2::ElementV1", calibStore, chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_Cspad2x2Element, 1, converter);

  // very special converter for Gsc16ai::DataV1, it needs two types of data
  converter = make_shared<Gsc16aiDataV1Cvt>("Gsc16ai::DataV1", configStore, chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_Gsc16aiData, 1, converter);

  // very special converter for Timepix::DataV1
  // Note that it makes group DataV2 as internally it converts DataV1 into DataV2
  converter = make_shared<TimepixDataV1Cvt>("Timepix::DataV2", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_TimepixData, 1, converter);

  // very special converter for Timepix::DataV2
  converter = make_shared<TimepixDataV2Cvt>("Timepix::DataV2", chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_TimepixData, 2, converter);

  // very special converter for OceanOptics::DataV1, it needs two types of data
  converter = make_shared<OceanOpticsDataV1Cvt>("OceanOptics::DataV1", configStore, chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_OceanOpticsData, 1, converter);

  // very special converter for Fli::FrameV1, it needs two types of data
  converter = make_shared<FliFrameV1Cvt<H5DataTypes::FliFrameV1> >("Fli::FrameV1", configStore, Pds::TypeId(Pds::TypeId::Id_FliConfig, 1), chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_FliFrame, 1, converter);

  // very special converter for Andor::FrameV1, it needs two types of data
  converter = make_shared<FliFrameV1Cvt<H5DataTypes::AndorFrameV1> >("Andor::FrameV1", configStore, Pds::TypeId(Pds::TypeId::Id_AndorConfig, 1), chunk_size, compression);
  ::registerCvt(m_cvtMap, Pds::TypeId::Id_AndorFrame, 1, converter);

}

} // namespace O2OTranslator
