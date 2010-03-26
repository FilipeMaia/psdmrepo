//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataPhaseCavity...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataPhaseCavity.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fFitTime1)
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fFitTime2)
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fCharge1)
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fCharge2)

  PyGetSetDef getset[] = {
    {"fFitTime1", fFitTime1, 0, "PV name: UND:R02:IOC:16:BAT:FitTime1, in pico-seconds", 0},
    {"fFitTime2", fFitTime2, 0, "PV name: UND:R02:IOC:16:BAT:FitTime2,in pico-seconds", 0},
    {"fCharge1",  fCharge1,  0, "PV name: UND:R02:IOC:16:BAT:Charge1,in pico-seconds", 0},
    {"fCharge2",  fCharge2,  0, "PV name: UND:R02:IOC:16:BAT:Charge2,in pico-seconds", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataPhaseCavity class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataPhaseCavity::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataPhaseCavity", module );
}
