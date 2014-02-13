//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Python module pdsdata...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

//-----------------
// C/C++ Headers --
//-----------------
#include "python/Python.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "BldInfo.h"
#include "ClockTime.h"
#include "Damage.h"
#include "Dgram.h"
#include "DetInfo.h"
#include "Env.h"
#include "Exception.h"
#include "Level.h"
#include "L1AcceptEnv.h"
#include "ProcInfo.h"
#include "Sequence.h"
#include "TimeStamp.h"
#include "TransitionId.h"
#include "TypeId.h"
#include "Xtc.h"
#include "XtcEmbedded.h"
#include "XtcFileIterator.h"
#include "XtcIterator.h"

#include "io/XtcFilter.h"
#include "io/XtcFilterTypeId.h"

#include "types/acqiris/ConfigV1.h"
#include "types/acqiris/DataDescV1.h"
#include "types/acqiris/DataDescV1Elem.h"
#include "types/acqiris/HorizV1.h"
#include "types/acqiris/TdcAuxIO.h"
#include "types/acqiris/TdcChannel.h"
#include "types/acqiris/TdcConfigV1.h"
#include "types/acqiris/TdcDataV1.h"
#include "types/acqiris/TdcDataV1Channel.h"
#include "types/acqiris/TdcDataV1Common.h"
#include "types/acqiris/TdcDataV1Marker.h"
#include "types/acqiris/TdcVetoIO.h"
#include "types/acqiris/TimestampV1.h"
#include "types/acqiris/TrigV1.h"
#include "types/acqiris/VertV1.h"

#include "types/alias/ConfigV1.h"
#include "types/alias/SrcAlias.h"

#include "types/andor/ConfigV1.h"
#include "types/andor/FrameV1.h"

#include "types/bld/BldDataAcqADCV1.h"
#include "types/bld/BldDataEBeamV0.h"
#include "types/bld/BldDataEBeamV1.h"
#include "types/bld/BldDataEBeamV2.h"
#include "types/bld/BldDataEBeamV3.h"
#include "types/bld/BldDataEBeamV4.h"
#include "types/bld/BldDataFEEGasDetEnergy.h"
#include "types/bld/BldDataGMDV0.h"
#include "types/bld/BldDataGMDV1.h"
#include "types/bld/BldDataIpimbV0.h"
#include "types/bld/BldDataIpimbV1.h"
#include "types/bld/BldDataPhaseCavity.h"
#include "types/bld/BldDataPimV1.h"
#include "types/bld/BldDataSpectrometerV0.h"

#include "types/camera/FrameCoord.h"
#include "types/camera/FrameFexConfigV1.h"
#include "types/camera/FrameV1.h"
#include "types/camera/TwoDGaussianV1.h"

#include "types/control/ConfigV1.h"
#include "types/control/ConfigV2.h"
#include "types/control/ConfigV3.h"
#include "types/control/PVControl.h"
#include "types/control/PVLabel.h"
#include "types/control/PVMonitor.h"

#include "types/cspad/ConfigV1.h"
#include "types/cspad/ConfigV1QuadReg.h"
#include "types/cspad/ConfigV2.h"
#include "types/cspad/ConfigV2QuadReg.h"
#include "types/cspad/ConfigV3.h"
#include "types/cspad/ConfigV3QuadReg.h"
#include "types/cspad/ConfigV4.h"
#include "types/cspad/ConfigV5.h"
#include "types/cspad/CsPadDigitalPotsCfg.h"
#include "types/cspad/CsPadGainMapCfg.h"
#include "types/cspad/CsPadProtectionSystemThreshold.h"
#include "types/cspad/CsPadReadOnlyCfg.h"
#include "types/cspad/DataV1.h"
#include "types/cspad/DataV2.h"
#include "types/cspad/ElementV1.h"
#include "types/cspad/ElementV2.h"

#include "types/cspad2x2/ConfigV1.h"
#include "types/cspad2x2/ConfigV1QuadReg.h"
#include "types/cspad2x2/ConfigV2.h"
#include "types/cspad2x2/ConfigV2QuadReg.h"
#include "types/cspad2x2/CsPad2x2DigitalPotsCfg.h"
#include "types/cspad2x2/CsPad2x2GainMapCfg.h"
#include "types/cspad2x2/CsPad2x2ReadOnlyCfg.h"
#include "types/cspad2x2/CsPadProtectionSystemThreshold.h"
#include "types/cspad2x2/ElementV1.h"

#include "types/encoder/ConfigV1.h"
#include "types/encoder/ConfigV2.h"
#include "types/encoder/DataV1.h"
#include "types/encoder/DataV2.h"

#include "types/epics/EpicsModule.h"

#include "types/epix/AsicConfigV1.h"
#include "types/epix/ConfigV1.h"
#include "types/epix/ElementV1.h"

#include "types/epixsampler/ConfigV1.h"
#include "types/epixsampler/ElementV1.h"

#include "types/evr/ConfigV1.h"
#include "types/evr/ConfigV2.h"
#include "types/evr/ConfigV3.h"
#include "types/evr/ConfigV4.h"
#include "types/evr/ConfigV5.h"
#include "types/evr/ConfigV6.h"
#include "types/evr/ConfigV7.h"
#include "types/evr/DataV3.h"
#include "types/evr/DataV3_FIFOEvent.h"
#include "types/evr/EventCodeV3.h"
#include "types/evr/EventCodeV4.h"
#include "types/evr/EventCodeV5.h"
#include "types/evr/EventCodeV6.h"
#include "types/evr/IOChannel.h"
#include "types/evr/IOConfigV1.h"
#include "types/evr/OutputMap.h"
#include "types/evr/OutputMapV2.h"
#include "types/evr/PulseConfig.h"
#include "types/evr/PulseConfigV3.h"
#include "types/evr/SequencerConfigV1.h"
#include "types/evr/SequencerEntry.h"
#include "types/evr/SrcConfigV1.h"
#include "types/evr/SrcEventCode.h"

#include "types/fccd/FccdConfigV1.h"
#include "types/fccd/FccdConfigV2.h"

#include "types/fli/ConfigV1.h"
#include "types/fli/FrameV1.h"

#include "types/gsc16ai/ConfigV1.h"
#include "types/gsc16ai/DataV1.h"

#include "types/imp/ConfigV1.h"
#include "types/imp/ElementV1.h"
#include "types/imp/LaneStatus.h"
#include "types/imp/Sample.h"

#include "types/ipimb/ConfigV1.h"
#include "types/ipimb/ConfigV2.h"
#include "types/ipimb/DataV1.h"
#include "types/ipimb/DataV2.h"

#include "types/l3t/ConfigV1.h"
#include "types/l3t/DataV1.h"

#include "types/lusi/DiodeFexConfigV1.h"
#include "types/lusi/DiodeFexConfigV2.h"
#include "types/lusi/DiodeFexV1.h"
#include "types/lusi/IpmFexConfigV1.h"
#include "types/lusi/IpmFexConfigV2.h"
#include "types/lusi/IpmFexV1.h"
#include "types/lusi/PimImageConfigV1.h"

#include "types/oceanoptics/ConfigV1.h"
#include "types/oceanoptics/DataV1.h"

#include "types/opal1k/ConfigV1.h"

#include "types/orca/ConfigV1.h"

#include "types/partition/ConfigV1.h"
#include "types/partition/Source.h"

#include "types/pnCCD/ConfigV1.h"
#include "types/pnCCD/ConfigV2.h"
#include "types/pnCCD/FrameV1.h"
#include "types/pnCCD/FramesV1.h"

#include "types/princeton/ConfigV1.h"
#include "types/princeton/ConfigV2.h"
#include "types/princeton/ConfigV3.h"
#include "types/princeton/ConfigV4.h"
#include "types/princeton/ConfigV5.h"
#include "types/princeton/FrameV1.h"
#include "types/princeton/FrameV2.h"
#include "types/princeton/InfoV1.h"

#include "types/pulnix/TM6740ConfigV1.h"
#include "types/pulnix/TM6740ConfigV2.h"

#include "types/quartz/ConfigV1.h"

#include "types/rayonix/ConfigV1.h"
#include "types/rayonix/ConfigV2.h"

#include "types/timepix/ConfigV1.h"
#include "types/timepix/ConfigV2.h"
#include "types/timepix/ConfigV3.h"
#include "types/timepix/DataV1.h"
#include "types/timepix/DataV2.h"

#include "types/usdusb/ConfigV1.h"
#include "types/usdusb/DataV1.h"

#define PDSDATA_IMPORT_ARRAY
#include "pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  void registerType( PyObject* this_module, const char* name, PyTypeObject* type ) {
    if ( PyType_Ready( type ) < 0 ) return;

    Py_INCREF( type );
    PyModule_AddObject( this_module, (char*)name, (PyObject*) type );
  }

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

// Module entry point
extern "C"
PyMODINIT_FUNC init_pdsdata()
{
  // Initialize the module
  PyObject* this_module = Py_InitModule3( "_pdsdata", 0, "The Python module for XTC" );

  PyObject* excType = pypdsdata::exceptionType();
  Py_INCREF( excType );
  PyModule_AddObject( this_module, "Error", excType );

  PyObject* module = 0;

  module = Py_InitModule3( "_pdsdata.xtc", 0, "The Python wrapper module for pdsdata/xtc" );
  pypdsdata::BldInfo::initType( module );
  pypdsdata::ClockTime::initType( module );
  pypdsdata::Damage::initType( module );
  pypdsdata::DetInfo::initType( module );
  pypdsdata::Dgram::initType( module );
  pypdsdata::Env::initType( module );
  ::registerType( module, "Level", pypdsdata::Level::typeObject() );
  pypdsdata::L1AcceptEnv::initType( module );
  pypdsdata::ProcInfo::initType( module );
  pypdsdata::Sequence::initType( module );
  pypdsdata::TimeStamp::initType( module );
  ::registerType( module, "TransitionId", pypdsdata::TransitionId::typeObject() );
  pypdsdata::TypeId::initType( module );
  pypdsdata::Xtc::initType( module );
  pypdsdata::XtcEmbedded::initType( module );
  ::registerType( module, "XtcFileIterator", pypdsdata::XtcFileIterator::typeObject() );
  ::registerType( module, "XtcIterator", pypdsdata::XtcIterator::typeObject() );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "xtc", module );

  // initialize data types, each in its own module
  module = Py_InitModule3( "_pdsdata.acqiris", 0, "The Python wrapper module for pdsdata/acqiris" );
  pypdsdata::Acqiris::ConfigV1::initType( module );
  pypdsdata::Acqiris::DataDescV1::initType( module );
  pypdsdata::Acqiris::DataDescV1Elem::initType( module );
  pypdsdata::Acqiris::HorizV1::initType( module );
  pypdsdata::Acqiris::TdcAuxIO::initType( module );
  pypdsdata::Acqiris::TdcChannel::initType( module );
  pypdsdata::Acqiris::TdcConfigV1::initType( module );
  pypdsdata::Acqiris::TdcDataV1::initType( module );
  pypdsdata::Acqiris::TdcDataV1Channel::initType( module );
  pypdsdata::Acqiris::TdcDataV1Common::initType( module );
  pypdsdata::Acqiris::TdcDataV1Marker::initType( module );
  pypdsdata::Acqiris::TdcVetoIO::initType( module );
  pypdsdata::Acqiris::TimestampV1::initType( module );
  pypdsdata::Acqiris::TrigV1::initType( module );
  pypdsdata::Acqiris::VertV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "acqiris", module );

  module = Py_InitModule3( "_pdsdata.alias", 0, "The Python wrapper module for pdsdata/alias" );
  pypdsdata::Alias::ConfigV1::initType( module );
  pypdsdata::Alias::SrcAlias::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "alias", module );

  module = Py_InitModule3( "_pdsdata.andor", 0, "The Python wrapper module for pdsdata/andor" );
  pypdsdata::Andor::ConfigV1::initType( module );
  pypdsdata::Andor::FrameV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "andor", module );

  module = Py_InitModule3( "_pdsdata.bld", 0, "The Python wrapper module for pdsdata/bld" );
  pypdsdata::Bld::BldDataAcqADCV1::initType( module );
  pypdsdata::Bld::BldDataEBeamV0::initType( module );
  pypdsdata::Bld::BldDataEBeamV1::initType( module );
  pypdsdata::Bld::BldDataEBeamV2::initType( module );
  pypdsdata::Bld::BldDataEBeamV3::initType( module );
  pypdsdata::Bld::BldDataEBeamV4::initType( module );
  pypdsdata::Bld::BldDataFEEGasDetEnergy::initType( module );
  pypdsdata::Bld::BldDataGMDV0::initType( module );
  pypdsdata::Bld::BldDataGMDV1::initType( module );
  pypdsdata::Bld::BldDataIpimbV0::initType( module );
  pypdsdata::Bld::BldDataIpimbV1::initType( module );
  pypdsdata::Bld::BldDataPhaseCavity::initType( module );
  pypdsdata::Bld::BldDataPimV1::initType( module );
  pypdsdata::Bld::BldDataSpectrometerV0::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "bld", module );

  module = Py_InitModule3( "_pdsdata.camera", 0, "The Python wrapper module for pdsdata/camera" );
  pypdsdata::Camera::FrameCoord::initType( module );
  pypdsdata::Camera::FrameFexConfigV1::initType( module );
  pypdsdata::Camera::FrameV1::initType( module );
  pypdsdata::Camera::TwoDGaussianV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "camera", module );

  module = Py_InitModule3( "_pdsdata.control", 0, "The Python wrapper module for pdsdata/control" );
  pypdsdata::ControlData::ConfigV1::initType( module );
  pypdsdata::ControlData::ConfigV2::initType( module );
  pypdsdata::ControlData::ConfigV3::initType( module );
  pypdsdata::ControlData::PVControl::initType( module );
  pypdsdata::ControlData::PVLabel::initType( module );
  pypdsdata::ControlData::PVMonitor::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "control", module );

  module = Py_InitModule3( "_pdsdata.cspad", 0, "The Python wrapper module for pdsdata/cspad" );
  pypdsdata::CsPad::ConfigV1::initType( module );
  pypdsdata::CsPad::ConfigV1QuadReg::initType( module );
  pypdsdata::CsPad::ConfigV2::initType( module );
  pypdsdata::CsPad::ConfigV2QuadReg::initType( module );
  pypdsdata::CsPad::ConfigV3::initType( module );
  pypdsdata::CsPad::ConfigV3QuadReg::initType( module );
  pypdsdata::CsPad::ConfigV4::initType( module );
  pypdsdata::CsPad::ConfigV5::initType( module );
  pypdsdata::CsPad::CsPadDigitalPotsCfg::initType( module );
  pypdsdata::CsPad::CsPadGainMapCfg::initType( module );
  pypdsdata::CsPad::CsPadProtectionSystemThreshold::initType( module );
  pypdsdata::CsPad::CsPadReadOnlyCfg::initType( module );
  pypdsdata::CsPad::DataV1::initType( module );
  pypdsdata::CsPad::DataV2::initType( module );
  pypdsdata::CsPad::ElementV1::initType( module );
  pypdsdata::CsPad::ElementV2::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "cspad", module );

  module = Py_InitModule3( "_pdsdata.cspad2x2", 0, "The Python wrapper module for pdsdata/cspad2x2" );
  pypdsdata::CsPad2x2::ConfigV1::initType( module );
  pypdsdata::CsPad2x2::ConfigV1QuadReg::initType( module );
  pypdsdata::CsPad2x2::ConfigV2::initType( module );
  pypdsdata::CsPad2x2::ConfigV2QuadReg::initType( module );
  pypdsdata::CsPad2x2::CsPad2x2DigitalPotsCfg::initType( module );
  pypdsdata::CsPad2x2::CsPad2x2GainMapCfg::initType( module );
  pypdsdata::CsPad2x2::CsPadProtectionSystemThreshold::initType( module );
  pypdsdata::CsPad2x2::CsPad2x2ReadOnlyCfg::initType( module );
  pypdsdata::CsPad2x2::ElementV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "cspad2x2", module );

  module = Py_InitModule3( "_pdsdata.encoder", 0, "The Python wrapper module for pdsdata/encoder" );
  pypdsdata::Encoder::ConfigV1::initType( module );
  pypdsdata::Encoder::ConfigV2::initType( module );
  pypdsdata::Encoder::DataV1::initType( module );
  pypdsdata::Encoder::DataV2::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "encoder", module );

  // very special epics module
  module = pypdsdata::Epics::EpicsModule::getModule();
  Py_INCREF( module );
  PyModule_AddObject( this_module, "epics", module );

  module = Py_InitModule3( "_pdsdata.epix", 0, "The Python wrapper module for pdsdata/epix" );
  pypdsdata::Epix::AsicConfigV1::initType( module );
  pypdsdata::Epix::ConfigV1::initType( module );
  pypdsdata::Epix::ElementV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "epix", module );

  module = Py_InitModule3( "_pdsdata.epixsampler", 0, "The Python wrapper module for pdsdata/epixsampler" );
  pypdsdata::EpixSampler::ConfigV1::initType( module );
  pypdsdata::EpixSampler::ElementV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "epixsampler", module );

  module = Py_InitModule3( "_pdsdata.evr", 0, "The Python wrapper module for pdsdata/evr" );
  pypdsdata::EvrData::ConfigV1::initType( module );
  pypdsdata::EvrData::ConfigV2::initType( module );
  pypdsdata::EvrData::ConfigV3::initType( module );
  pypdsdata::EvrData::ConfigV4::initType( module );
  pypdsdata::EvrData::ConfigV5::initType( module );
  pypdsdata::EvrData::ConfigV6::initType( module );
  pypdsdata::EvrData::ConfigV7::initType( module );
  pypdsdata::EvrData::DataV3::initType( module );
  pypdsdata::EvrData::DataV3_FIFOEvent::initType( module );
  pypdsdata::EvrData::EventCodeV3::initType( module );
  pypdsdata::EvrData::EventCodeV4::initType( module );
  pypdsdata::EvrData::EventCodeV5::initType( module );
  pypdsdata::EvrData::EventCodeV6::initType( module );
  pypdsdata::EvrData::IOChannel::initType( module );
  pypdsdata::EvrData::IOConfigV1::initType( module );
  pypdsdata::EvrData::OutputMap::initType( module );
  pypdsdata::EvrData::OutputMapV2::initType( module );
  pypdsdata::EvrData::PulseConfig::initType( module );
  pypdsdata::EvrData::PulseConfigV3::initType( module );
  pypdsdata::EvrData::SequencerConfigV1::initType( module );
  pypdsdata::EvrData::SequencerEntry::initType( module );
  pypdsdata::EvrData::SrcConfigV1::initType( module );
  pypdsdata::EvrData::SrcEventCode::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "evr", module );

  module = Py_InitModule3( "_pdsdata.fccd", 0, "The Python wrapper module for pdsdata/fccd" );
  pypdsdata::FCCD::FccdConfigV1::initType( module );
  pypdsdata::FCCD::FccdConfigV2::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "fccd", module );

  module = Py_InitModule3( "_pdsdata.fli", 0, "The Python wrapper module for pdsdata/fli" );
  pypdsdata::Fli::ConfigV1::initType( module );
  pypdsdata::Fli::FrameV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "fli", module );

  module = Py_InitModule3( "_pdsdata.gsc16ai", 0, "The Python wrapper module for pdsdata/gsc16ai" );
  pypdsdata::Gsc16ai::ConfigV1::initType( module );
  pypdsdata::Gsc16ai::DataV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "gsc16ai", module );

  module = Py_InitModule3( "_pdsdata.imp", 0, "The Python wrapper module for pdsdata/imp" );
  pypdsdata::Imp::ConfigV1::initType( module );
  pypdsdata::Imp::ElementV1::initType( module );
  pypdsdata::Imp::LaneStatus::initType( module );
  pypdsdata::Imp::Sample::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "imp", module );

  module = Py_InitModule3( "_pdsdata.ipimb", 0, "The Python wrapper module for pdsdata/ipimb" );
  pypdsdata::Ipimb::ConfigV1::initType( module );
  pypdsdata::Ipimb::ConfigV2::initType( module );
  pypdsdata::Ipimb::DataV1::initType( module );
  pypdsdata::Ipimb::DataV2::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "ipimb", module );

  module = Py_InitModule3( "_pdsdata.l3t", 0, "The Python wrapper module for pdsdata/l3t" );
  pypdsdata::L3T::ConfigV1::initType( module );
  pypdsdata::L3T::DataV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "l3t", module );

  module = Py_InitModule3( "_pdsdata.lusi", 0, "The Python wrapper module for pdsdata/lusi" );
  pypdsdata::Lusi::DiodeFexConfigV1::initType( module );
  pypdsdata::Lusi::DiodeFexConfigV2::initType( module );
  pypdsdata::Lusi::DiodeFexV1::initType( module );
  pypdsdata::Lusi::IpmFexConfigV1::initType( module );
  pypdsdata::Lusi::IpmFexConfigV2::initType( module );
  pypdsdata::Lusi::IpmFexV1::initType( module );
  pypdsdata::Lusi::PimImageConfigV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "lusi", module );

  module = Py_InitModule3( "_pdsdata.oceanoptics", 0, "The Python wrapper module for pdsdata/oceanoptics" );
  pypdsdata::OceanOptics::ConfigV1::initType( module );
  pypdsdata::OceanOptics::DataV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "oceanoptics", module );

  module = Py_InitModule3( "_pdsdata.opal1k", 0, "The Python wrapper module for pdsdata/opal1k" );
  pypdsdata::Opal1k::ConfigV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "opal1k", module );

  module = Py_InitModule3( "_pdsdata.orca", 0, "The Python wrapper module for pdsdata/orca" );
  pypdsdata::Orca::ConfigV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "orca", module );

  module = Py_InitModule3( "_pdsdata.partition", 0, "The Python wrapper module for pdsdata/partition" );
  pypdsdata::Partition::ConfigV1::initType( module );
  pypdsdata::Partition::Source::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "partition", module );

  module = Py_InitModule3( "_pdsdata.pnccd", 0, "The Python wrapper module for pdsdata/pnCCD" );
  pypdsdata::PNCCD::ConfigV1::initType( module );
  pypdsdata::PNCCD::ConfigV2::initType( module );
  pypdsdata::PNCCD::FrameV1::initType( module );
  pypdsdata::PNCCD::FramesV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "pnccd", module );

  module = Py_InitModule3( "_pdsdata.princeton", 0, "The Python wrapper module for pdsdata/princeton" );
  pypdsdata::Princeton::ConfigV1::initType( module );
  pypdsdata::Princeton::ConfigV2::initType( module );
  pypdsdata::Princeton::ConfigV3::initType( module );
  pypdsdata::Princeton::ConfigV4::initType( module );
  pypdsdata::Princeton::ConfigV5::initType( module );
  pypdsdata::Princeton::FrameV1::initType( module );
  pypdsdata::Princeton::FrameV2::initType( module );
  pypdsdata::Princeton::InfoV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "princeton", module );

  module = Py_InitModule3( "_pdsdata.pulnix", 0, "The Python wrapper module for pdsdata/pulnix" );
  pypdsdata::Pulnix::TM6740ConfigV1::initType( module );
  pypdsdata::Pulnix::TM6740ConfigV2::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "pulnix", module );

  module = Py_InitModule3( "_pdsdata.quartz", 0, "The Python wrapper module for pdsdata/quartz" );
  pypdsdata::Quartz::ConfigV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "quartz", module );

  module = Py_InitModule3( "_pdsdata.rayonix", 0, "The Python wrapper module for pdsdata/rayonix" );
  pypdsdata::Rayonix::ConfigV1::initType( module );
  pypdsdata::Rayonix::ConfigV2::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "rayonix", module );

  module = Py_InitModule3( "_pdsdata.timepix", 0, "The Python wrapper module for pdsdata/timepix" );
  pypdsdata::Timepix::ConfigV1::initType( module );
  pypdsdata::Timepix::ConfigV2::initType( module );
  pypdsdata::Timepix::ConfigV3::initType( module );
  pypdsdata::Timepix::DataV1::initType( module );
  pypdsdata::Timepix::DataV2::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "timepix", module );

  module = Py_InitModule3( "_pdsdata.usdusb", 0, "The Python wrapper module for pdsdata/usdusb" );
  pypdsdata::UsdUsb::ConfigV1::initType( module );
  pypdsdata::UsdUsb::DataV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "usdusb", module );

  module = Py_InitModule3( "_pdsdata.io", 0, "The Python module for several I/O related classes" );
  pypdsdata::XtcFilter::initType( module );
  pypdsdata::XtcFilterTypeId::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "io", module );

  // import NumPy
  import_array();

}
