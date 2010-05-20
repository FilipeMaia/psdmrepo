//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataObjectFactory...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataObjectFactory.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Xtc.h"

#include "types/acqiris/ConfigV1.h"
#include "types/acqiris/DataDescV1.h"

#include "types/bld/BldDataFEEGasDetEnergy.h"
#include "types/bld/BldDataEBeamV0.h"
#include "types/bld/BldDataEBeam.h"
#include "types/bld/BldDataPhaseCavity.h"

#include "types/camera/FrameFexConfigV1.h"
#include "types/camera/FrameV1.h"
#include "types/camera/TwoDGaussianV1.h"

#include "types/control/ConfigV1.h"

#include "types/encoder/ConfigV1.h"
#include "types/encoder/DataV1.h"

#include "types/epics/EpicsModule.h"

#include "types/evr/ConfigV1.h"
#include "types/evr/ConfigV2.h"
#include "types/evr/ConfigV3.h"
#include "types/evr/DataV3.h"

#include "types/fccd/FccdConfigV1.h"

#include "types/ipimb/ConfigV1.h"
#include "types/ipimb/DataV1.h"

#include "types/opal1k/ConfigV1.h"

#include "types/pnCCD/ConfigV1.h"
#include "types/pnCCD/FrameV1.h"

#include "types/princeton/ConfigV1.h"
#include "types/princeton/FrameV1.h"

#include "types/pulnix/TM6740ConfigV1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // "destructor" for cloned xtc
  void buf_dealloc(Pds::Xtc* xtc) {
    delete [] (char*)xtc;
  }

  // make a separate copy of the xtc object not sharing same buffer
  pypdsdata::Xtc* cloneXtc( const Pds::Xtc& xtc )
  {
    size_t size = xtc.extent ;
    char* newbuf = new char[size];
    const char* oldbuf = (const char*)&xtc;
    std::copy( oldbuf, oldbuf+size, newbuf);

    return pypdsdata::Xtc::PyObject_FromPds( (Pds::Xtc*)newbuf, 0, size, ::buf_dealloc );
  }

  template <typename T, int Version>
  inline 
  PyObject* xtc2obj(const Pds::Xtc& xtc, PyObject* parent) {
    if( Version < 0 or xtc.contains.version() == unsigned(Version) ) {
      return T::PyObject_FromXtc(xtc, parent); 
    }
    return 0;
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pypdsdata {

/**
 * Factory method which creates Python objects from XTC
 */
PyObject*
DataObjectFactory::makeObject( const Pds::Xtc& xtc, PyObject* parent )
{
  PyObject* obj = 0;
  Xtc* clone = 0;
  switch ( xtc.contains.id() ) {

  case Pds::TypeId::Any :
  case Pds::TypeId::Id_Xtc :
    // handled in Xtc class
    break;

  case Pds::TypeId::Id_Frame :
    if ( not obj ) obj = xtc2obj<Camera::FrameV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_AcqWaveform :
    if ( not obj ) obj = xtc2obj<Acqiris::DataDescV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_AcqConfig :
    if ( not obj ) obj = xtc2obj<Acqiris::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_TwoDGaussian :
    if ( not obj ) obj = xtc2obj<Camera::TwoDGaussianV1, 1>(xtc, parent);
    break;

  case Pds::TypeId::Id_Opal1kConfig :
    if ( not obj ) obj = xtc2obj<Opal1k::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_FrameFexConfig :
    if ( not obj ) obj = xtc2obj<Camera::FrameFexConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_EvrConfig :
    if ( not obj ) obj = xtc2obj<EvrData::ConfigV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<EvrData::ConfigV2, 2>(xtc, parent);
    if ( not obj ) obj = xtc2obj<EvrData::ConfigV3, 3>(xtc, parent);
    break ;

  case Pds::TypeId::Id_TM6740Config :
    if ( not obj ) obj = xtc2obj<Pulnix::TM6740ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_ControlConfig :
    if ( not obj ) obj = xtc2obj<ControlData::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_pnCCDframe :
    if ( not obj ) obj = xtc2obj<PNCCD::FrameV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_pnCCDconfig :
    if ( not obj ) obj = xtc2obj<PNCCD::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_Epics :
    clone = ::cloneXtc( xtc );
    obj = EpicsModule::PyObject_FromXtc(*clone->m_obj, clone);
    Py_CLEAR(clone);
    break ;

  case Pds::TypeId::Id_FEEGasDetEnergy :
    // NOTE: does not seem to care about versions
    if ( not obj ) obj = xtc2obj<BldDataFEEGasDetEnergy, -1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_EBeam :
    if ( not obj ) obj = xtc2obj<BldDataEBeamV0, 0>(xtc, parent);
    if ( not obj ) obj = xtc2obj<BldDataEBeam, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_PhaseCavity :
    // NOTE: does not seem to care about versions
    if ( not obj ) obj = xtc2obj<BldDataPhaseCavity, -1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_PrincetonFrame :
    if ( not obj ) obj = xtc2obj<Princeton::FrameV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_PrincetonConfig :
    if ( not obj ) obj = xtc2obj<Princeton::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_EvrData :
    if ( not obj ) obj = xtc2obj<EvrData::DataV3, 3>(xtc, parent);
    break ;

  case Pds::TypeId::Id_FccdConfig :
    if ( not obj ) obj = xtc2obj<FCCD::FccdConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_IpimbData :
    if ( not obj ) obj = xtc2obj<Ipimb::DataV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_IpimbConfig :
    if ( not obj ) obj = xtc2obj<Ipimb::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_EncoderData :
    if ( not obj ) obj = xtc2obj<Encoder::DataV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_EncoderConfig :
    if ( not obj ) obj = xtc2obj<Encoder::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::NumberOf :
    // just to make compiler shut up about this special unhandled enum
    break;
  }

  if ( not obj ) {
    PyErr_Format(PyExc_NotImplementedError, "Error: DataObjectFactory unsupported type %s", Pds::TypeId::name(xtc.contains.id()) );
  }

  return obj ;
}

} // namespace pypdsdata
