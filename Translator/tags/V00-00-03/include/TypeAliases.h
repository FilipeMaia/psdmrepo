/* Need to generate this file from DDL - to get the list of 
   comments
*/

#ifndef TRANSLATOR_TYPEALIAS_MAP_H
#define TRANSLATOR_TYPEALIAS_MAP_H

#include <string>
#include <map>
#include <set>
#include <typeinfo>
#include "PSEvt/TypeInfoUtils.h"

namespace Translator {

  /**
   * @ingroup TypeAlias
   *
   * @brief class providing aliases to refer to sets of Psana types.
   * 
   * For example, 'CsPad' is an alias for all of the Config and Data CsPad types.  
   * The intention it to provide aliases that make it easy to filter type
   * without using the C++ type names.  The groupings are listed below.  They
   * are generated from the DDL, so they may not be perfect.  For instance
   * if one needs to distinguish between CsPad::DataV1 and CsPad::DataV2, 
   * you cannot use these aliases.
   * 
   * Alias list:
    <pre>     
    AcqTdc                      Psana::Acqiris::TdcConfigV1, Psana::Acqiris::TdcDataV1
    AcqWaveform                 Psana::Acqiris::ConfigV1, Psana::Acqiris::DataDescV1
    Andor                       Psana::Andor::ConfigV1, Psana::Andor::FrameV1
    ControlConfig               Psana::ControlData::ConfigV1, Psana::ControlData::ConfigV2
    Cspad                       Psana::CsPad::ConfigV1, Psana::CsPad::ConfigV2, Psana::CsPad::ConfigV3, Psana::CsPad::ConfigV4, Psana::CsPad::ConfigV5, Psana::CsPad::DataV1, Psana::CsPad::DataV2
    Cspad2x2                    Psana::CsPad2x2::ConfigV1, Psana::CsPad2x2::ConfigV2, Psana::CsPad2x2::ElementV1
    DiodeFex                    Psana::Lusi::DiodeFexConfigV1, Psana::Lusi::DiodeFexConfigV2, Psana::Lusi::DiodeFexV1
    EBeam                       Psana::Bld::BldDataEBeamV0, Psana::Bld::BldDataEBeamV1, Psana::Bld::BldDataEBeamV2, Psana::Bld::BldDataEBeamV3
    Encoder                     Psana::Encoder::ConfigV1, Psana::Encoder::ConfigV2, Psana::Encoder::DataV1, Psana::Encoder::DataV2
    EpicsConfig                 Psana::Epics::ConfigV1
    Evr                         Psana::EvrData::ConfigV1, Psana::EvrData::ConfigV2, Psana::EvrData::ConfigV3, Psana::EvrData::ConfigV4, Psana::EvrData::ConfigV5, Psana::EvrData::ConfigV6, Psana::EvrData::ConfigV7, Psana::EvrData::DataV3
    EvrIOConfig                 Psana::EvrData::IOConfigV1
    FEEGasDetEnergy             Psana::Bld::BldDataFEEGasDetEnergy
    FccdConfig                  Psana::FCCD::FccdConfigV1, Psana::FCCD::FccdConfigV2
    Fli                         Psana::Fli::ConfigV1, Psana::Fli::FrameV1
    Frame                       Psana::Camera::FrameV1
    FrameFccdConfig             Psana::Camera::FrameFccdConfigV1
    FrameFexConfig              Psana::Camera::FrameFexConfigV1
    GMD                         Psana::Bld::BldDataGMDV0, Psana::Bld::BldDataGMDV1
    Gsc16ai                     Psana::Gsc16ai::ConfigV1, Psana::Gsc16ai::DataV1
    Imp                         Psana::Imp::ConfigV1, Psana::Imp::ElementV1
    Ipimb                       Psana::Ipimb::ConfigV1, Psana::Ipimb::ConfigV2, Psana::Ipimb::DataV1, Psana::Ipimb::DataV2
    IpmFex                      Psana::Lusi::IpmFexConfigV1, Psana::Lusi::IpmFexConfigV2, Psana::Lusi::IpmFexV1
    OceanOptics                 Psana::OceanOptics::ConfigV1, Psana::OceanOptics::DataV1
    Opal1kConfig                Psana::Opal1k::ConfigV1
    OrcaConfig                  Psana::Orca::ConfigV1
    PhaseCavity                 Psana::Bld::BldDataPhaseCavity
    PimImageConfig              Psana::Lusi::PimImageConfigV1
    Princeton                   Psana::Princeton::ConfigV1, Psana::Princeton::ConfigV2, Psana::Princeton::ConfigV3, Psana::Princeton::ConfigV4, Psana::Princeton::ConfigV5, Psana::Princeton::FrameV1, Psana::Princeton::FrameV2
    PrincetonInfo               Psana::Princeton::InfoV1
    QuartzConfig                Psana::Quartz::ConfigV1
    SharedIpimb                 Psana::Bld::BldDataIpimbV0, Psana::Bld::BldDataIpimbV1
    SharedPim                   Psana::Bld::BldDataPimV1
    TM6740Config                Psana::Pulnix::TM6740ConfigV1, Psana::Pulnix::TM6740ConfigV2
    Timepix                     Psana::Timepix::ConfigV1, Psana::Timepix::ConfigV2, Psana::Timepix::ConfigV3, Psana::Timepix::DataV1, Psana::Timepix::DataV2
    TwoDGaussian                Psana::Camera::TwoDGaussianV1
    UsdUsb                      Psana::UsdUsb::ConfigV1, Psana::UsdUsb::DataV1
    pnCCD                       Psana::PNCCD::ConfigV1, Psana::PNCCD::ConfigV2, Psana::PNCCD::FramesV1, Psana::PNCCD::FullFrameV1
   </pre>
   */

class TypeAliases {
public:
  TypeAliases();
  typedef std::set<const std::type_info *, PSEvt::TypeInfoUtils::lessTypeInfoPtr> TypeInfoSet;
  typedef std::map<std::string, TypeInfoSet > Alias2TypesMap;
  typedef std::map<const std::type_info *, std::string,  PSEvt::TypeInfoUtils::lessTypeInfoPtr > Type2AliasMap;

  const std::set<std::string> & aliases() { return m_aliasKeys; }
  const Alias2TypesMap & alias2TypesMap(){ return m_alias2TypesMap; }
  const Type2AliasMap & type2AliasMap(){ return m_type2AliasMap; }
private:
  std::set<std::string> m_aliasKeys;
  Alias2TypesMap m_alias2TypesMap;
  Type2AliasMap m_type2AliasMap;
};

} // namespace

#endif
