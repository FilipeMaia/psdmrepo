//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataEBeamV3.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iomanip>

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
      { "EbeamChargeDamage",    Pds::BldDataEBeamV3::EbeamChargeDamage },
      { "EbeamL3EnergyDamage",  Pds::BldDataEBeamV3::EbeamL3EnergyDamage },
      { "EbeamLTUPosXDamage",   Pds::BldDataEBeamV3::EbeamLTUPosXDamage },
      { "EbeamLTUPosYDamage",   Pds::BldDataEBeamV3::EbeamLTUPosYDamage },
      { "EbeamLTUAngXDamage",   Pds::BldDataEBeamV3::EbeamLTUAngXDamage },
      { "EbeamLTUAngYDamage",   Pds::BldDataEBeamV3::EbeamLTUAngYDamage },
      { "EbeamPkCurrBC2Damage", Pds::BldDataEBeamV3::EbeamPkCurrBC2Damage },
      { "EbeamEnergyBC2Damage", Pds::BldDataEBeamV3::EbeamEnergyBC2Damage },
      { "EbeamPkCurrBC1Damage", Pds::BldDataEBeamV3::EbeamPkCurrBC1Damage },
      { "EbeamEnergyBC1Damage", Pds::BldDataEBeamV3::EbeamEnergyBC1Damage },
      { 0, 0 }
  };
  pypdsdata::EnumType damageMaskEnum ( "DamageMask", damageMaskEnumValues );

  // methods
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, uDamageMask)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamCharge)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamL3Energy)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamLTUPosX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamLTUPosY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamLTUAngX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamLTUAngY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamPkCurrBC2)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamEnergyBC2)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamPkCurrBC1)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV3, fEbeamEnergyBC1)

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
    {"fEbeamEnergyBC2", fEbeamEnergyBC2, 0, "floating number, beam position (related to energy), in mm", 0},
    {"fEbeamPkCurrBC1", fEbeamPkCurrBC1, 0, "floating number, in Amps", 0},
    {"fEbeamEnergyBC1", fEbeamEnergyBC1, 0, "floating number, beam position (related to energy), in mm", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataEBeamV3 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataEBeamV3::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "DamageMask", damageMaskEnum.type() );

  BaseType::initType( "BldDataEBeamV3", module );
}

void
pypdsdata::BldDataEBeamV3::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(DamageMask=" << std::showbase << std::hex << m_obj->uDamageMask << std::dec
        << ", Charge=" << m_obj->fEbeamCharge
        << ", L3Energy=" << m_obj->fEbeamL3Energy << ", ...)";
  }
}
