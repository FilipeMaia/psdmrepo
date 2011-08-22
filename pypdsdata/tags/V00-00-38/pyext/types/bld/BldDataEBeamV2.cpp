//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataEBeamV2.h"

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
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV2, uDamageMask)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV2, fEbeamCharge)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV2, fEbeamL3Energy)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV2, fEbeamLTUPosX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV2, fEbeamLTUPosY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV2, fEbeamLTUAngX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV2, fEbeamLTUAngY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV2, fEbeamPkCurrBC2)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV2, fEbeamEnergyBC2)
  PyObject* _repr( PyObject *self );

  PyGetSetDef getset[] = {
    {"uDamageMask",    uDamageMask,    0, "", 0},
    {"fEbeamCharge",   fEbeamCharge,   0, "in nC", 0},
    {"fEbeamL3Energy", fEbeamL3Energy, 0, "in MeV", 0},
    {"fEbeamLTUPosX",  fEbeamLTUPosX,  0, "in mm", 0},
    {"fEbeamLTUPosY",  fEbeamLTUPosY,  0, "in mm", 0},
    {"fEbeamLTUAngX",  fEbeamLTUAngX,  0, "in mrad", 0},
    {"fEbeamLTUAngY",  fEbeamLTUAngY,  0, "in mrad", 0},
    {"fEbeamPkCurrBC2", fEbeamPkCurrBC2, 0, "in Amps", 0},
    {"fEbeamEnergyBC2", fEbeamEnergyBC2, 0, "in MeV", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataEBeamV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataEBeamV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "BldDataEBeamV2", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataEBeamV2* pdsObj = pypdsdata::BldDataEBeamV2::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[64];
  snprintf( buf, sizeof buf, "BldDataEBeamV2(Charge=%f, L3Energy=%f, ...)",
            pdsObj->fEbeamCharge, pdsObj->fEbeamL3Energy );
  return PyString_FromString( buf );
}

}
