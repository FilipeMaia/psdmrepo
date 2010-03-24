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

#include "types/epics/EpicsModule.h"

#include "types/evr/ConfigV1.h"
#include "types/evr/ConfigV2.h"

#include "types/opal1k/ConfigV1.h"

#include "types/pnCCD/ConfigV1.h"
#include "types/pnCCD/FrameV1.h"

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
    if ( xtc.contains.version() == 1 ) {
      obj = Camera::FrameV1::PyObject_FromXtc(xtc, parent);
    }
    break ;

  case Pds::TypeId::Id_AcqWaveform :
    if ( xtc.contains.version() == 1 ) {
      obj = Acqiris::DataDescV1::PyObject_FromXtc(xtc, parent);
    }
    break ;

  case Pds::TypeId::Id_AcqConfig :
    if ( xtc.contains.version() == 1 ) {
      obj = Acqiris::ConfigV1::PyObject_FromXtc(xtc, parent);
    }
    break ;

  case Pds::TypeId::Id_TwoDGaussian :
    if ( xtc.contains.version() == 1 ) {
      obj = Camera::TwoDGaussianV1::PyObject_FromXtc(xtc, parent);
    }
    break;

  case Pds::TypeId::Id_Opal1kConfig :
    if ( xtc.contains.version() == 1 ) {
      obj = Opal1k::ConfigV1::PyObject_FromXtc(xtc, parent);
    }
    break ;

  case Pds::TypeId::Id_FrameFexConfig :
    if ( xtc.contains.version() == 1 ) {
      obj = Camera::FrameFexConfigV1::PyObject_FromXtc(xtc, parent);
    }
    break ;

  case Pds::TypeId::Id_EvrConfig :
    if ( xtc.contains.version() == 1 ) {
      obj = EvrData::ConfigV1::PyObject_FromXtc(xtc, parent);
    } else if ( xtc.contains.version() == 2 ) {
      obj = EvrData::ConfigV2::PyObject_FromXtc(xtc, parent);
    }
    break ;

  case Pds::TypeId::Id_TM6740Config :
    if ( xtc.contains.version() == 1 ) {
      obj = Pulnix::TM6740ConfigV1::PyObject_FromXtc(xtc, parent);
    }
    break ;

  case Pds::TypeId::Id_ControlConfig :
    if ( xtc.contains.version() == 1 ) {
      obj = ControlData::ConfigV1::PyObject_FromXtc(xtc, parent);
    }
    break ;

  case Pds::TypeId::Id_pnCCDframe :
    if ( xtc.contains.version() == 1 ) {
      obj = PNCCD::FrameV1::PyObject_FromXtc(xtc, parent);
    }
    break ;

  case Pds::TypeId::Id_pnCCDconfig :
    if ( xtc.contains.version() == 1 ) {
      obj = PNCCD::ConfigV1::PyObject_FromXtc(xtc, parent);
    }
    break ;

  case Pds::TypeId::Id_Epics :
    clone = ::cloneXtc( xtc );
    obj = EpicsModule::PyObject_FromXtc(*clone->m_obj, clone);
    Py_CLEAR(clone);
    break ;

  case Pds::TypeId::Id_FEEGasDetEnergy :
    // NOTE: does not seem to care about versions
    obj = BldDataFEEGasDetEnergy::PyObject_FromXtc(xtc, parent);
    break ;

  case Pds::TypeId::Id_EBeam :
    if ( xtc.contains.version() == 0 ) {
      obj = BldDataEBeamV0::PyObject_FromXtc(xtc, parent);
    } else {
      obj = BldDataEBeam::PyObject_FromXtc(xtc, parent);
    }
    break ;

  case Pds::TypeId::Id_PhaseCavity :
    // NOTE: does not seem to care about versions
    obj = BldDataPhaseCavity::PyObject_FromXtc(xtc, parent);
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
