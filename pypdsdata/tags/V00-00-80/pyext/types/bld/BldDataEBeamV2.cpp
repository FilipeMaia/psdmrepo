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
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum damageMaskEnumValues[] = {
      { "EbeamChargeDamage",    Pds::BldDataEBeamV2::EbeamChargeDamage },
      { "EbeamL3EnergyDamage",  Pds::BldDataEBeamV2::EbeamL3EnergyDamage },
      { "EbeamLTUPosXDamage",   Pds::BldDataEBeamV2::EbeamLTUPosXDamage },
      { "EbeamLTUPosYDamage",   Pds::BldDataEBeamV2::EbeamLTUPosYDamage },
      { "EbeamLTUAngXDamage",   Pds::BldDataEBeamV2::EbeamLTUAngXDamage },
      { "EbeamLTUAngYDamage",   Pds::BldDataEBeamV2::EbeamLTUAngYDamage },
      { "EbeamPkCurrBC2Damage", Pds::BldDataEBeamV2::EbeamPkCurrBC2Damage },
      { "EbeamEnergyBC2Damage", Pds::BldDataEBeamV2::EbeamEnergyBC2Damage },
      { 0, 0 }
  };
  pypdsdata::EnumType damageMaskEnum ( "DamageMask", damageMaskEnumValues );

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

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"uDamageMask",    uDamageMask,    0, "integer bit mask, see :py:class:`DamageMask` for individual bits meaning", 0},
    {"fEbeamCharge",   fEbeamCharge,   0, "floating number, in nC", 0},
    {"fEbeamL3Energy", fEbeamL3Energy, 0, "floating number, in MeV", 0},
    {"fEbeamLTUPosX",  fEbeamLTUPosX,  0, "floating number, in mm", 0},
    {"fEbeamLTUPosY",  fEbeamLTUPosY,  0, "floating number, in mm", 0},
    {"fEbeamLTUAngX",  fEbeamLTUAngX,  0, "floating number, in mrad", 0},
    {"fEbeamLTUAngY",  fEbeamLTUAngY,  0, "floating number, in mrad", 0},
    {"fEbeamPkCurrBC2", fEbeamPkCurrBC2, 0, "floating number, in Amps", 0},
    {"fEbeamEnergyBC2", fEbeamEnergyBC2, 0, "floating number, in MeV", 0},
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

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "DamageMask", damageMaskEnum.type() );

  BaseType::initType( "BldDataEBeamV2", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataEBeamV2* pdsObj = pypdsdata::BldDataEBeamV2::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[64];
  snprintf( buf, sizeof buf, "BldDataEBeamV2(Charge=%g, L3Energy=%g, ...)",
            pdsObj->fEbeamCharge, pdsObj->fEbeamL3Energy );
  return PyString_FromString( buf );
}

}
