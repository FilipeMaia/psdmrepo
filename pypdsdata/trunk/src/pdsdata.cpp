//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class pypdsdata...
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
#include "Python.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "BldInfo.h"
#include "ClockTime.h"
#include "Damage.h"
#include "Datagram.h"
#include "DetInfo.h"
#include "Level.h"
#include "ProcInfo.h"
#include "Sequence.h"
#include "TimeStamp.h"
#include "TransitionId.h"
#include "TypeId.h"
#include "Xtc.h"
#include "XtcFileIterator.h"
#include "XtcIterator.h"

#include "types/camera/FrameCoord.h"
#include "types/camera/FrameFexConfigV1.h"
#include "types/camera/FrameV1.h"

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
PyMODINIT_FUNC initpdsdata()
{
  // Initialize the module
  PyObject* this_module = Py_InitModule3( "pdsdata", 0, "The Python module for XTC" );

  ::registerType( this_module, "BldInfo", pypdsdata::BldInfo::typeObject() );
  ::registerType( this_module, "ClockTime", pypdsdata::ClockTime::typeObject() );
  ::registerType( this_module, "Damage", pypdsdata::Damage::typeObject() );
  ::registerType( this_module, "Datagram", pypdsdata::Datagram::typeObject() );
  ::registerType( this_module, "DetInfo", pypdsdata::DetInfo::typeObject() );
  ::registerType( this_module, "Level", pypdsdata::Level::typeObject() );
  ::registerType( this_module, "ProcInfo", pypdsdata::ProcInfo::typeObject() );
  ::registerType( this_module, "Sequence", pypdsdata::Sequence::typeObject() );
  ::registerType( this_module, "TimeStamp", pypdsdata::TimeStamp::typeObject() );
  ::registerType( this_module, "TransitionId", pypdsdata::TransitionId::typeObject() );
  ::registerType( this_module, "TypeId", pypdsdata::TypeId::typeObject() );
  ::registerType( this_module, "Xtc", pypdsdata::Xtc::typeObject() );
  ::registerType( this_module, "XtcFileIterator", pypdsdata::XtcFileIterator::typeObject() );
  ::registerType( this_module, "XtcIterator", pypdsdata::XtcIterator::typeObject() );

  // initialize data types, but do not register them
  PyType_Ready( pypdsdata::Camera::FrameCoord::typeObject() );
  PyType_Ready( pypdsdata::Camera::FrameFexConfigV1::typeObject() );
  PyType_Ready( pypdsdata::Camera::FrameV1::typeObject() );

  // import NumPy
  import_array();

}
