//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class HdfGroupName...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psddl_hdf2psana/HdfGroupName.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <map>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/ProcInfo.hh"
#include "psddl_hdf2psana/Exceptions.h"
#include "PSEvt/Source.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  std::map<std::string, Pds::TypeId> initTypeIdMap()
  {
    std::map<std::string, Pds::TypeId> mapping;
    // Initialize mapping
#define ADD2MAP(NAME, TYPEID, VERSION) mapping.insert(std::make_pair(std::string(NAME), Pds::TypeId(Pds::TypeId::TYPEID, VERSION)))
    ADD2MAP("Acqiris::ConfigV1",            Id_AcqConfig,       1);
    ADD2MAP("Acqiris::AcqirisTdcConfigV1",  Id_AcqTdcConfig,    1);
    ADD2MAP("Opal1k::ConfigV1",             Id_Opal1kConfig,    1);
    ADD2MAP("Pulnix::TM6740ConfigV1",       Id_TM6740Config,    1);
    ADD2MAP("Pulnix::TM6740ConfigV2",       Id_TM6740Config,    2);
    ADD2MAP("Camera::FrameFexConfigV1",     Id_FrameFexConfig,  1);
    ADD2MAP("EvrData::ConfigV1",            Id_EvrConfig,       1);
    ADD2MAP("EvrData::ConfigV2",            Id_EvrConfig,       2);
    ADD2MAP("EvrData::ConfigV3",            Id_EvrConfig,       3);
    ADD2MAP("EvrData::ConfigV4",            Id_EvrConfig,       4);
    ADD2MAP("EvrData::ConfigV5",            Id_EvrConfig,       5);
    ADD2MAP("EvrData::ConfigV6",            Id_EvrConfig,       6);
    ADD2MAP("EvrData::IOConfigV1",          Id_EvrIOConfig,     1);
    ADD2MAP("ControlData::ConfigV1",        Id_ControlConfig,   1);
    ADD2MAP("PNCCD::ConfigV1",              Id_pnCCDconfig,     1);
    ADD2MAP("PNCCD::ConfigV2",              Id_pnCCDconfig,     2);
    ADD2MAP("Princeton::ConfigV1",          Id_PrincetonConfig, 1);
    ADD2MAP("Princeton::ConfigV2",          Id_PrincetonConfig, 2);
    ADD2MAP("FCCD::FccdConfigV1",           Id_FccdConfig,      1);
    ADD2MAP("FCCD::FccdConfigV2",           Id_FccdConfig,      2);
    ADD2MAP("Ipimb::ConfigV1",              Id_IpimbConfig,     1);
    ADD2MAP("Ipimb::ConfigV2",              Id_IpimbConfig,     2);
    ADD2MAP("Encoder::ConfigV1",            Id_EncoderConfig,   1);
    ADD2MAP("Encoder::ConfigV2",            Id_EncoderConfig,   2);
    ADD2MAP("Lusi::DiodeFexConfigV1",       Id_DiodeFexConfig,  1);
    ADD2MAP("Lusi::DiodeFexConfigV2",       Id_DiodeFexConfig,  2);
    ADD2MAP("Lusi::IpmFexConfigV1",         Id_IpmFexConfig,    1);
    ADD2MAP("Lusi::IpmFexConfigV2",         Id_IpmFexConfig,    2);
    ADD2MAP("Lusi::PimImageConfigV1",       Id_PimImageConfig,  1);
    ADD2MAP("CsPad::ConfigV1",              Id_CspadConfig,     1);
    ADD2MAP("CsPad::ConfigV2",              Id_CspadConfig,     2);
    ADD2MAP("CsPad::ConfigV3",              Id_CspadConfig,     3);
    ADD2MAP("Gsc16ai::ConfigV1",            Id_Gsc16aiConfig,   1);
    ADD2MAP("Timepix::ConfigV1",            Id_TimepixConfig,   1);
    ADD2MAP("Camera::TwoDGaussianV1",       Id_TwoDGaussian,    1);
    ADD2MAP("Bld::BldDataFEEGasDetEnergy",  Id_FEEGasDetEnergy, 1);
    ADD2MAP("Bld::BldDataEBeamV0",          Id_EBeam,           0);
    ADD2MAP("Bld::BldDataEBeam",            Id_EBeam,           1);
    ADD2MAP("Bld::BldDataEBeamV1",          Id_EBeam,           1);
    ADD2MAP("Bld::BldDataEBeamV2",          Id_EBeam,           2);
    ADD2MAP("Bld::BldDataEBeamV3",          Id_EBeam,           3);
    ADD2MAP("Bld::BldDataIpimb",            Id_SharedIpimb,     0);
    ADD2MAP("Bld::BldDataIpimbV0",          Id_SharedIpimb,     0);
    ADD2MAP("Bld::BldDataIpimbV1",          Id_SharedIpimb,     1);
    ADD2MAP("Bld::BldDataPhaseCavity",      Id_PhaseCavity,     0);
    ADD2MAP("Bld::BldDataPimV1",            Id_SharedPim,       1);
    ADD2MAP("Encoder::DataV1",              Id_EncoderData,     1);
    ADD2MAP("Encoder::DataV2",              Id_EncoderData,     2);
    ADD2MAP("Ipimb::DataV1",                Id_IpimbData,       1);
    ADD2MAP("Ipimb::DataV2",                Id_IpimbData,       2);
    ADD2MAP("Camera::FrameV1",              Id_Frame,           1);
    ADD2MAP("Acqiris::DataDescV1",          Id_AcqWaveform,     1);
    ADD2MAP("Acqiris::TdcDataV1",           Id_AcqTdcData,      1);
    ADD2MAP("EvrData::DataV3",              Id_EvrData,         3);
    ADD2MAP("PNCCD::FrameV1",               Id_pnCCDframe,      1);
    ADD2MAP("Princeton::FrameV1",           Id_PrincetonFrame,  1);
    ADD2MAP("Princeton::InfoV1",            Id_PrincetonInfo,   1);
    ADD2MAP("Epics::EpicsPv",               Id_Epics,           1);
    ADD2MAP("Lusi::DiodeFexV1",             Id_DiodeFex,        1);
    ADD2MAP("Lusi::IpmFexV1",               Id_IpmFex,          1);
    ADD2MAP("CsPad::ElementV1",             Id_CspadElement,    1);
    ADD2MAP("CsPad::ElementV2",             Id_CspadElement,    2);
    ADD2MAP("CsPad::MiniElementV1",         Id_Cspad2x2Element, 1);
    ADD2MAP("Gsc16ai::DataV1",              Id_Gsc16aiData,     1);
    ADD2MAP("Timepix::DataV1",              Id_TimepixData,     1);
#undef ADD2MAP
    return mapping;
  }

  // Maps group name to TypeId
  std::map<std::string, Pds::TypeId> name2typeId = initTypeIdMap();


}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psddl_hdf2psana {

// Get TypeId for the group name
Pds::TypeId
HdfGroupName::nameToTypeId(const std::string& name)
{

  std::map<std::string, Pds::TypeId>::const_iterator it = ::name2typeId.find(name);
  if (it != ::name2typeId.end()) return it->second;

  // throw on non-existing TypeId
  throw ExceptionGroupTypeIdName(ERR_LOC, name);
}

// Get Src from group name
Pds::Src
HdfGroupName::nameToSource(const std::string& name)
{
  if (name == "Control") {

    // Special case for ProcInfo at Control level
    return Pds::ProcInfo(Pds::Level::Control, 0, 0);

  } else {

    // parse it for existing PSEvt parser
    try {
      PSEvt::Source src(name);
      PSEvt::Source::SrcMatch srcm = src.srcMatch(PSEvt::AliasMap());
      if (not srcm.isExact()) {
        // parsed OK but contains wildcards?
        throw ExceptionGroupSourceName(ERR_LOC, name);
      }
      return srcm.src();
    } catch (const std::exception& ex) {
      throw ExceptionGroupSourceName(ERR_LOC, name);
    }

  }
}

} // namespace psddl_hdf2psana
