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
#include "types/acqiris/TdcConfigV1.h"
#include "types/acqiris/TdcDataV1.h"

#include "types/bld/BldDataFEEGasDetEnergy.h"
#include "types/bld/BldDataEBeamV0.h"
#include "types/bld/BldDataEBeamV1.h"
#include "types/bld/BldDataEBeamV2.h"
#include "types/bld/BldDataEBeamV3.h"
#include "types/bld/BldDataIpimbV0.h"
#include "types/bld/BldDataIpimbV1.h"
#include "types/bld/BldDataPhaseCavity.h"
#include "types/bld/BldDataPimV1.h"

#include "types/camera/FrameFexConfigV1.h"
#include "types/camera/FrameV1.h"
#include "types/camera/TwoDGaussianV1.h"

#include "types/control/ConfigV1.h"

#include "types/cspad/ConfigV1.h"
#include "types/cspad/ConfigV2.h"
#include "types/cspad/ConfigV3.h"
#include "types/cspad/ConfigV4.h"
#include "types/cspad/ElementV1.h"
#include "types/cspad/ElementV2.h"

#include "types/cspad2x2/ConfigV1.h"
#include "types/cspad2x2/ElementV1.h"

#include "types/encoder/ConfigV1.h"
#include "types/encoder/ConfigV2.h"
#include "types/encoder/DataV1.h"
#include "types/encoder/DataV2.h"

#include "types/epics/EpicsModule.h"
#include "types/epics/ConfigV1.h"

#include "types/evr/ConfigV1.h"
#include "types/evr/ConfigV2.h"
#include "types/evr/ConfigV3.h"
#include "types/evr/ConfigV4.h"
#include "types/evr/ConfigV5.h"
#include "types/evr/ConfigV6.h"
#include "types/evr/DataV3.h"
#include "types/evr/IOConfigV1.h"

#include "types/fccd/FccdConfigV1.h"
#include "types/fccd/FccdConfigV2.h"

#include "types/fli/ConfigV1.h"
#include "types/fli/FrameV1.h"

#include "types/gsc16ai/ConfigV1.h"
#include "types/gsc16ai/DataV1.h"

#include "types/ipimb/ConfigV1.h"
#include "types/ipimb/ConfigV2.h"
#include "types/ipimb/DataV1.h"
#include "types/ipimb/DataV2.h"

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

#include "types/pnCCD/ConfigV1.h"
#include "types/pnCCD/ConfigV2.h"
#include "types/pnCCD/FrameV1.h"

#include "types/princeton/ConfigV1.h"
#include "types/princeton/ConfigV2.h"
#include "types/princeton/ConfigV3.h"
#include "types/princeton/FrameV1.h"
#include "types/princeton/InfoV1.h"

#include "types/pulnix/TM6740ConfigV1.h"
#include "types/pulnix/TM6740ConfigV2.h"

#include "types/quartz/ConfigV1.h"

#include "types/timepix/ConfigV1.h"
#include "types/timepix/ConfigV2.h"
#include "types/timepix/DataV1.h"
#include "types/timepix/DataV2.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // "destructor" for cloned xtc
  void buf_dealloc(Pds::Xtc* xtc) {
    PyMem_Free(xtc);
  }

  // make a separate copy of the xtc object not sharing same buffer, return 0 if memory allocation fails
  pypdsdata::Xtc* cloneXtc( const Pds::Xtc& xtc )
  {
    size_t size = xtc.extent ;
    char* newbuf = (char*)PyMem_Malloc(size);
    if (not newbuf) return 0;
    const char* oldbuf = (const char*)&xtc;
    std::copy(oldbuf, oldbuf+size, newbuf);

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
    if ( not obj ) obj = xtc2obj<EvrData::ConfigV4, 4>(xtc, parent);
    if ( not obj ) obj = xtc2obj<EvrData::ConfigV5, 5>(xtc, parent);
    if ( not obj ) obj = xtc2obj<EvrData::ConfigV6, 6>(xtc, parent);
    break ;

  case Pds::TypeId::Id_TM6740Config :
    if ( not obj ) obj = xtc2obj<Pulnix::TM6740ConfigV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<Pulnix::TM6740ConfigV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_ControlConfig :
    if ( not obj ) obj = xtc2obj<ControlData::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_pnCCDframe :
    if ( not obj ) obj = xtc2obj<PNCCD::FrameV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_pnCCDconfig :
    if ( not obj ) obj = xtc2obj<PNCCD::ConfigV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<PNCCD::ConfigV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_Epics :
    if (xtc.sizeofPayload() == 0) {
      // some strange kind of damage where Epics data has 0 size
      Py_RETURN_NONE;
    } else {
      if (Xtc* clone = ::cloneXtc(xtc)) {
        obj = EpicsModule::PyObject_FromXtc(*clone->m_obj, clone);
        Py_CLEAR(clone);
      } else {
        PyErr_Format(PyExc_MemoryError, "Error: failed to allocate buffer memory for XTC clone");
      }
    }
    break ;

  case Pds::TypeId::Id_FEEGasDetEnergy :
    // NOTE: does not seem to care about versions
    if ( not obj ) obj = xtc2obj<BldDataFEEGasDetEnergy, -1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_EBeam :
    if ( not obj ) obj = xtc2obj<BldDataEBeamV0, 0>(xtc, parent);
    if ( not obj ) obj = xtc2obj<BldDataEBeamV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<BldDataEBeamV2, 2>(xtc, parent);
    if ( not obj ) obj = xtc2obj<BldDataEBeamV3, 3>(xtc, parent);
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
    if ( not obj ) obj = xtc2obj<Princeton::ConfigV2, 2>(xtc, parent);
    if ( not obj ) obj = xtc2obj<Princeton::ConfigV3, 3>(xtc, parent);
    break ;

  case Pds::TypeId::Id_EvrData :
    if ( not obj ) obj = xtc2obj<EvrData::DataV3, 3>(xtc, parent);
    break ;

  case Pds::TypeId::Id_FrameFccdConfig :
    // there is no sensible data definition for this type, XTC class is empty
    break ;

  case Pds::TypeId::Id_FccdConfig :
    if ( not obj ) obj = xtc2obj<FCCD::FccdConfigV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<FCCD::FccdConfigV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_IpimbData :
    if ( not obj ) obj = xtc2obj<Ipimb::DataV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<Ipimb::DataV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_IpimbConfig :
    if ( not obj ) obj = xtc2obj<Ipimb::ConfigV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<Ipimb::ConfigV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_EncoderData :
    if ( not obj ) obj = xtc2obj<Encoder::DataV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<Encoder::DataV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_EncoderConfig :
    if ( not obj ) obj = xtc2obj<Encoder::ConfigV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<Encoder::ConfigV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_EvrIOConfig :
    if ( not obj ) obj = xtc2obj<EvrData::IOConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_PrincetonInfo :
    if ( not obj ) obj = xtc2obj<Princeton::InfoV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_CspadElement :
    if ( not obj ) obj = xtc2obj<CsPad::ElementV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<CsPad::ElementV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_CspadConfig :
    if ( not obj ) obj = xtc2obj<CsPad::ConfigV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<CsPad::ConfigV2, 2>(xtc, parent);
    if ( not obj ) obj = xtc2obj<CsPad::ConfigV3, 3>(xtc, parent);
    if ( not obj ) obj = xtc2obj<CsPad::ConfigV4, 4>(xtc, parent);
    break ;

  case Pds::TypeId::Id_IpmFexConfig :
    if ( not obj ) obj = xtc2obj<Lusi::IpmFexConfigV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<Lusi::IpmFexConfigV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_IpmFex :
    if ( not obj ) obj = xtc2obj<Lusi::IpmFexV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_DiodeFexConfig :
    if ( not obj ) obj = xtc2obj<Lusi::DiodeFexConfigV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<Lusi::DiodeFexConfigV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_DiodeFex :
    if ( not obj ) obj = xtc2obj<Lusi::DiodeFexV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_PimImageConfig :
    if ( not obj ) obj = xtc2obj<Lusi::PimImageConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_SharedIpimb :
    if ( not obj ) obj = xtc2obj<BldDataIpimbV0, 0>(xtc, parent);
    if ( not obj ) obj = xtc2obj<BldDataIpimbV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_AcqTdcConfig :
    if ( not obj ) obj = xtc2obj<Acqiris::TdcConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_AcqTdcData :
    if ( not obj ) obj = xtc2obj<Acqiris::TdcDataV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_Index :
    break;

  case Pds::TypeId::Id_XampsConfig :
    break;

  case Pds::TypeId::Id_XampsElement :
    break;

  case Pds::TypeId::Id_Cspad2x2Element :
    if ( not obj ) obj = xtc2obj<CsPad2x2::ElementV1, 1>(xtc, parent);
    break;

  case Pds::TypeId::Id_SharedPim :
    if ( not obj ) obj = xtc2obj<BldDataPimV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_Cspad2x2Config :
    if ( not obj ) obj = xtc2obj<CsPad2x2::ConfigV1, 1>(xtc, parent);
    break;

  case Pds::TypeId::Id_FexampConfig :
    break;

  case Pds::TypeId::Id_FexampElement :
    break;

  case Pds::TypeId::Id_Gsc16aiConfig :
    if ( not obj ) obj = xtc2obj<Gsc16ai::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_Gsc16aiData :
    if ( not obj ) obj = xtc2obj<Gsc16ai::DataV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_PhasicsConfig :
    break;

  case Pds::TypeId::Id_TimepixConfig :
    if ( not obj ) obj = xtc2obj<Timepix::ConfigV1, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<Timepix::ConfigV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_TimepixData :
    // very special conversion for V1, Timepix::DataV2 knows how to handle
    // both V1 and V2
    if ( not obj ) obj = xtc2obj<Timepix::DataV2, 1>(xtc, parent);
    if ( not obj ) obj = xtc2obj<Timepix::DataV2, 2>(xtc, parent);
    break ;

  case Pds::TypeId::Id_CspadCompressedElement :
    break;

  case Pds::TypeId::Id_OceanOpticsConfig :
    //if ( not obj ) obj = xtc2obj<OceanOptics::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_OceanOpticsData :
    //if ( not obj ) obj = xtc2obj<OceanOptics::DataV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_EpicsConfig :
    if ( not obj ) obj = xtc2obj<Epics::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_FliConfig :
    if ( not obj ) obj = xtc2obj<Fli::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_FliFrame :
    if ( not obj ) obj = xtc2obj<Fli::FrameV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::Id_QuartzConfig :
    if ( not obj ) obj = xtc2obj<Quartz::ConfigV1, 1>(xtc, parent);
    break ;

  case Pds::TypeId::NumberOf :
    // just to make compiler shut up about this special unhandled enum
    break;
  }

  if ( not obj ) {
    PyErr_Format(PyExc_NotImplementedError, "Error: DataObjectFactory unsupported type %s_V%d",
                 Pds::TypeId::name(xtc.contains.id()), xtc.contains.version() );
  }

  return obj ;
}

} // namespace pypdsdata
