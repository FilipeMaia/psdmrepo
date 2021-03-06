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
#include "Exception.h"
#include "Level.h"
#include "ProcInfo.h"
#include "Sequence.h"
#include "TimeStamp.h"
#include "TransitionId.h"
#include "TypeId.h"
#include "Xtc.h"
#include "XtcFileIterator.h"
#include "XtcIterator.h"

#include "types/acqiris/ConfigV1.h"
#include "types/acqiris/DataDescV1.h"
#include "types/acqiris/HorizV1.h"
#include "types/acqiris/TimestampV1.h"
#include "types/acqiris/TrigV1.h"
#include "types/acqiris/VertV1.h"

#include "types/bld/BldDataEBeam.h"
#include "types/bld/BldDataEBeamV0.h"
#include "types/bld/BldDataFEEGasDetEnergy.h"
#include "types/bld/BldDataPhaseCavity.h"

#include "types/camera/FrameCoord.h"
#include "types/camera/FrameFexConfigV1.h"
#include "types/camera/FrameV1.h"
#include "types/camera/TwoDGaussianV1.h"

#include "types/control/ConfigV1.h"
#include "types/control/PVControl.h"
#include "types/control/PVMonitor.h"

#include "types/encoder/ConfigV1.h"
#include "types/encoder/DataV1.h"

#include "types/epics/EpicsModule.h"

#include "types/evr/ConfigV1.h"
#include "types/evr/ConfigV2.h"
#include "types/evr/ConfigV3.h"
#include "types/evr/ConfigV4.h"
#include "types/evr/DataV3.h"
#include "types/evr/DataV3_FIFOEvent.h"
#include "types/evr/EventCodeV3.h"
#include "types/evr/EventCodeV4.h"
#include "types/evr/IOChannel.h"
#include "types/evr/IOConfigV1.h"
#include "types/evr/OutputMap.h"
#include "types/evr/PulseConfig.h"
#include "types/evr/PulseConfigV3.h"

#include "types/fccd/FccdConfigV1.h"
#include "types/fccd/FccdConfigV2.h"

#include "types/ipimb/ConfigV1.h"
#include "types/ipimb/DataV1.h"

#include "types/opal1k/ConfigV1.h"

#include "types/pnCCD/ConfigV1.h"
#include "types/pnCCD/ConfigV2.h"
#include "types/pnCCD/FrameV1.h"

#include "types/princeton/ConfigV1.h"
#include "types/princeton/FrameV1.h"

#include "types/pulnix/TM6740ConfigV1.h"
#include "types/pulnix/TM6740ConfigV2.h"

#define PDSDATA_IMPORT_ARRAY
#import "pdsdata_numpy.h"

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

  module = Py_InitModule3( "_pdsdata.xtc", 0, "The Python module for pdsdata/acqiris" );
  pypdsdata::BldInfo::initType( module );
  pypdsdata::ClockTime::initType( module );
  pypdsdata::Damage::initType( module );
  pypdsdata::DetInfo::initType( module );
  pypdsdata::Dgram::initType( module );
  ::registerType( module, "Level", pypdsdata::Level::typeObject() );
  pypdsdata::ProcInfo::initType( module );
  pypdsdata::Sequence::initType( module );
  pypdsdata::TimeStamp::initType( module );
  ::registerType( module, "TransitionId", pypdsdata::TransitionId::typeObject() );
  pypdsdata::TypeId::initType( module );
  pypdsdata::Xtc::initType( module );
  ::registerType( module, "XtcFileIterator", pypdsdata::XtcFileIterator::typeObject() );
  ::registerType( module, "XtcIterator", pypdsdata::XtcIterator::typeObject() );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "xtc", module );

  // initialize data types, each in its own module
  module = Py_InitModule3( "_pdsdata.acqiris", 0, "The Python module for pdsdata/acqiris" );
  pypdsdata::Acqiris::ConfigV1::initType( module );
  pypdsdata::Acqiris::DataDescV1::initType( module );
  pypdsdata::Acqiris::HorizV1::initType( module );
  pypdsdata::Acqiris::TimestampV1::initType( module );
  pypdsdata::Acqiris::TrigV1::initType( module );
  pypdsdata::Acqiris::VertV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "acqiris", module );

  module = Py_InitModule3( "_pdsdata.bld", 0, "The Python module for pdsdata/bld" );
  pypdsdata::BldDataEBeam::initType( module );
  pypdsdata::BldDataEBeamV0::initType( module );
  pypdsdata::BldDataFEEGasDetEnergy::initType( module );
  pypdsdata::BldDataPhaseCavity::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "bld", module );

  module = Py_InitModule3( "_pdsdata.camera", 0, "The Python module for pdsdata/camera" );
  pypdsdata::Camera::FrameCoord::initType( module );
  pypdsdata::Camera::FrameFexConfigV1::initType( module );
  pypdsdata::Camera::FrameV1::initType( module );
  pypdsdata::Camera::TwoDGaussianV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "camera", module );

  module = Py_InitModule3( "_pdsdata.control", 0, "The Python module for pdsdata/control" );
  pypdsdata::ControlData::ConfigV1::initType( module );
  pypdsdata::ControlData::PVControl::initType( module );
  pypdsdata::ControlData::PVMonitor::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "control", module );

  module = Py_InitModule3( "_pdsdata.encoder", 0, "The Python module for pdsdata/encoder" );
  pypdsdata::Encoder::ConfigV1::initType( module );
  pypdsdata::Encoder::DataV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "encoder", module );

  // very special epics module
  module = pypdsdata::EpicsModule::getModule();
  Py_INCREF( module );
  PyModule_AddObject( this_module, "epics", module );

  module = Py_InitModule3( "_pdsdata.evr", 0, "The Python module for pdsdata/evr" );
  pypdsdata::EvrData::ConfigV1::initType( module );
  pypdsdata::EvrData::ConfigV2::initType( module );
  pypdsdata::EvrData::ConfigV3::initType( module );
  pypdsdata::EvrData::ConfigV4::initType( module );
  pypdsdata::EvrData::DataV3::initType( module );
  pypdsdata::EvrData::DataV3_FIFOEvent::initType( module );
  pypdsdata::EvrData::EventCodeV3::initType( module );
  pypdsdata::EvrData::EventCodeV4::initType( module );
  pypdsdata::EvrData::IOChannel::initType( module );
  pypdsdata::EvrData::IOConfigV1::initType( module );
  pypdsdata::EvrData::OutputMap::initType( module );
  pypdsdata::EvrData::PulseConfig::initType( module );
  pypdsdata::EvrData::PulseConfigV3::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "evr", module );

  module = Py_InitModule3( "_pdsdata.fccd", 0, "The Python module for pdsdata/fccd" );
  pypdsdata::FCCD::FccdConfigV1::initType( module );
  pypdsdata::FCCD::FccdConfigV2::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "fccd", module );

  module = Py_InitModule3( "_pdsdata.ipimb", 0, "The Python module for pdsdata/ipimb" );
  pypdsdata::Ipimb::ConfigV1::initType( module );
  pypdsdata::Ipimb::DataV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "ipimb", module );

  module = Py_InitModule3( "_pdsdata.opal1k", 0, "The Python module for pdsdata/opal1k" );
  pypdsdata::Opal1k::ConfigV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "opal1k", module );

  module = Py_InitModule3( "_pdsdata.pnccd", 0, "The Python module for pdsdata/pnCCD" );
  pypdsdata::PNCCD::ConfigV1::initType( module );
  pypdsdata::PNCCD::ConfigV2::initType( module );
  pypdsdata::PNCCD::FrameV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "pnccd", module );

  module = Py_InitModule3( "_pdsdata.princeton", 0, "The Python module for pdsdata/princeton" );
  pypdsdata::Princeton::ConfigV1::initType( module );
  pypdsdata::Princeton::FrameV1::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "princeton", module );

  module = Py_InitModule3( "_pdsdata.pulnix", 0, "The Python module for pdsdata/pulnix" );
  pypdsdata::Pulnix::TM6740ConfigV1::initType( module );
  pypdsdata::Pulnix::TM6740ConfigV2::initType( module );
  Py_INCREF( module );
  PyModule_AddObject( this_module, "pulnix", module );

  // import NumPy
  import_array();

}
