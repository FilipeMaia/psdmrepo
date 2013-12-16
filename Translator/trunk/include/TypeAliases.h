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
   * The most current alias list is generated in default_psana.cfg
   @verbatim
AcqTdc = include               # Psana::Acqiris::TdcConfigV1, Psana::Acqiris::TdcDataV1
AcqWaveform = include          # Psana::Acqiris::ConfigV1, Psana::Acqiris::DataDescV1
Alias = include                # Psana::Alias::ConfigV1
Andor = include                # Psana::Andor::ConfigV1, Psana::Andor::FrameV1
Control = include              # Psana::ControlData::ConfigV1, Psana::ControlData::ConfigV2, Psana::ControlData::ConfigV3
Cspad = include                # Psana::CsPad::ConfigV1, Psana::CsPad::ConfigV2, Psana::CsPad::ConfigV3, Psana::CsPad::ConfigV4, Psana::CsPad::ConfigV5, Psana::CsPad::DataV1, Psana::CsPad::DataV2
Cspad2x2 = include             # Psana::CsPad2x2::ConfigV1, Psana::CsPad2x2::ConfigV2, Psana::CsPad2x2::ElementV1
DiodeFex = include             # Psana::Lusi::DiodeFexConfigV1, Psana::Lusi::DiodeFexConfigV2, Psana::Lusi::DiodeFexV1
EBeam = include                # Psana::Bld::BldDataEBeamV0, Psana::Bld::BldDataEBeamV1, Psana::Bld::BldDataEBeamV2, Psana::Bld::BldDataEBeamV3, Psana::Bld::BldDataEBeamV4
Encoder = include              # Psana::Encoder::ConfigV1, Psana::Encoder::ConfigV2, Psana::Encoder::DataV1, Psana::Encoder::DataV2
Epics = include                # Psana::Epics::ConfigV1
EpixSampler = include          # Psana::EpixSampler::ConfigV1, Psana::EpixSampler::ElementV1
Evr = include                  # Psana::EvrData::ConfigV1, Psana::EvrData::ConfigV2, Psana::EvrData::ConfigV3, Psana::EvrData::ConfigV4, Psana::EvrData::ConfigV5, Psana::EvrData::ConfigV6, Psana::EvrData::ConfigV7, Psana::EvrData::DataV3
EvrIO = include                # Psana::EvrData::IOConfigV1
FEEGasDetEnergy = include      # Psana::Bld::BldDataFEEGasDetEnergy
Fccd = include                 # Psana::FCCD::FccdConfigV1, Psana::FCCD::FccdConfigV2
Fli = include                  # Psana::Fli::ConfigV1, Psana::Fli::FrameV1
Frame = include                # Psana::Camera::FrameV1
FrameFccd = include            # Psana::Camera::FrameFccdConfigV1
FrameFex = include             # Psana::Camera::FrameFexConfigV1
GMD = include                  # Psana::Bld::BldDataGMDV0, Psana::Bld::BldDataGMDV1
Gsc16ai = include              # Psana::Gsc16ai::ConfigV1, Psana::Gsc16ai::DataV1
Imp = include                  # Psana::Imp::ConfigV1, Psana::Imp::ElementV1
Ipimb = include                # Psana::Ipimb::ConfigV1, Psana::Ipimb::ConfigV2, Psana::Ipimb::DataV1, Psana::Ipimb::DataV2
IpmFex = include               # Psana::Lusi::IpmFexConfigV1, Psana::Lusi::IpmFexConfigV2, Psana::Lusi::IpmFexV1
L3T = include                  # Psana::L3T::ConfigV1, Psana::L3T::DataV1
OceanOptics = include          # Psana::OceanOptics::ConfigV1, Psana::OceanOptics::DataV1
Opal1k = include               # Psana::Opal1k::ConfigV1
Orca = include                 # Psana::Orca::ConfigV1
PhaseCavity = include          # Psana::Bld::BldDataPhaseCavity
PimImage = include             # Psana::Lusi::PimImageConfigV1
Princeton = include            # Psana::Princeton::ConfigV1, Psana::Princeton::ConfigV2, Psana::Princeton::ConfigV3, Psana::Princeton::ConfigV4, Psana::Princeton::ConfigV5, Psana::Princeton::FrameV1, Psana::Princeton::FrameV2
PrincetonInfo = include        # Psana::Princeton::InfoV1
Quartz = include               # Psana::Quartz::ConfigV1
Rayonix = include              # Psana::Rayonix::ConfigV1, Psana::Rayonix::ConfigV2
SharedAcqADC = include         # Psana::Bld::BldDataAcqADCV1
SharedIpimb = include          # Psana::Bld::BldDataIpimbV0, Psana::Bld::BldDataIpimbV1
SharedPim = include            # Psana::Bld::BldDataPimV1
Spectrometer = include         # Psana::Bld::BldDataSpectrometerV0
TM6740 = include               # Psana::Pulnix::TM6740ConfigV1, Psana::Pulnix::TM6740ConfigV2
Timepix = include              # Psana::Timepix::ConfigV1, Psana::Timepix::ConfigV2, Psana::Timepix::ConfigV3, Psana::Timepix::DataV1, Psana::Timepix::DataV2
TwoDGaussian = include         # Psana::Camera::TwoDGaussianV1
UsdUsb = include               # Psana::UsdUsb::ConfigV1, Psana::UsdUsb::DataV1
pnCCD = include                # Psana::PNCCD::ConfigV1, Psana::PNCCD::ConfigV2, Psana::PNCCD::FramesV1

# user types to translate from the event store
ndarray_types = include        # ndarray<int8_t,1>, ndarray<int8_t,2>, ndarray<int8_t,3>, ndarray<int8_t,4>, ndarray<int16_t,1>, ndarray<int16_t,2>, ndarray<int16_t,3>, ndarray<int16_t,4>, ndarray<int32_t,1>, ndarray<int32_t,2>, ndarray<int32_t,3>, ndarray<int32_t,4>, ndarray<int64_t,1>, ndarray<int64_t,2>, ndarray<int64_t,3>, ndarray<int64_t,4>, ndarray<uint8_t,1>, ndarray<uint8_t,2>, ndarray<uint8_t,3>, ndarray<uint8_t,4>, ndarray<uint16_t,1>, ndarray<uint16_t,2>, ndarray<uint16_t,3>, ndarray<uint16_t,4>, ndarray<uint32_t,1>, ndarray<uint32_t,2>, ndarray<uint32_t,3>, ndarray<uint32_t,4>, ndarray<uint64_t,1>, ndarray<uint64_t,2>, ndarray<uint64_t,3>, ndarray<uint64_t,4>, ndarray<float,1>, ndarray<float,2>, ndarray<float,3>, ndarray<float,4>, ndarray<double,1>, ndarray<double,2>, ndarray<double,3>, ndarray<double,4>
std_string = include           # std::string
@endverbatim
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
