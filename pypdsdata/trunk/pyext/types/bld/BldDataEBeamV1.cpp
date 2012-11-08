//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataEBeamV1.h"

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
      { "EbeamChargeDamage",    Pds::BldDataEBeamV1::EbeamChargeDamage },
      { "EbeamL3EnergyDamage",  Pds::BldDataEBeamV1::EbeamL3EnergyDamage },
      { "EbeamLTUPosXDamage",   Pds::BldDataEBeamV1::EbeamLTUPosXDamage },
      { "EbeamLTUPosYDamage",   Pds::BldDataEBeamV1::EbeamLTUPosYDamage },
      { "EbeamLTUAngXDamage",   Pds::BldDataEBeamV1::EbeamLTUAngXDamage },
      { "EbeamLTUAngYDamage",   Pds::BldDataEBeamV1::EbeamLTUAngYDamage },
      { "EbeamPkCurrBC2Damage", Pds::BldDataEBeamV1::EbeamPkCurrBC2Damage },
      { 0, 0 }
  };
  pypdsdata::EnumType damageMaskEnum ( "DamageMask", damageMaskEnumValues );

  // methods
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV1, uDamageMask)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV1, fEbeamCharge)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV1, fEbeamL3Energy)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV1, fEbeamLTUPosX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV1, fEbeamLTUPosY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV1, fEbeamLTUAngX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV1, fEbeamLTUAngY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV1, fEbeamPkCurrBC2)
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
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataEBeamV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataEBeamV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "DamageMask", damageMaskEnum.type() );

  BaseType::initType( "BldDataEBeamV1", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataEBeamV1* pdsObj = pypdsdata::BldDataEBeamV1::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[64];
  snprintf( buf, sizeof buf, "BldDataEBeamV1(Charge=%g, L3Energy=%g, ...)",
            pdsObj->fEbeamCharge, pdsObj->fEbeamL3Energy );
  return PyString_FromString( buf );
}

}
