//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV0...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataEBeamV0.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Exception.h"
#include "types/TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, uDamageMask)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamCharge)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamL3Energy)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamLTUPosX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamLTUPosY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamLTUAngX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamLTUAngY)

  PyGetSetDef getset[] = {
    {"uDamageMask",    uDamageMask,    0, "", 0},
    {"fEbeamCharge",   fEbeamCharge,   0, "in nC", 0},
    {"fEbeamL3Energy", fEbeamL3Energy, 0, "in MeV", 0},
    {"fEbeamLTUPosX",  fEbeamLTUPosX,  0, "in mm", 0},
    {"fEbeamLTUPosY",  fEbeamLTUPosY,  0, "in mm", 0},
    {"fEbeamLTUAngX",  fEbeamLTUAngX,  0, "in mrad", 0},
    {"fEbeamLTUAngY",  fEbeamLTUAngY,  0, "in mrad", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataEBeamV0 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataEBeamV0::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataEBeamV0", module );
}
