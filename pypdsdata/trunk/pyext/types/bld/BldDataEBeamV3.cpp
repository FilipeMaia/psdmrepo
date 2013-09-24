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
      { "EbeamChargeDamage",    Pds::Bld::BldDataEBeamV3::EbeamChargeDamage },
      { "EbeamL3EnergyDamage",  Pds::Bld::BldDataEBeamV3::EbeamL3EnergyDamage },
      { "EbeamLTUPosXDamage",   Pds::Bld::BldDataEBeamV3::EbeamLTUPosXDamage },
      { "EbeamLTUPosYDamage",   Pds::Bld::BldDataEBeamV3::EbeamLTUPosYDamage },
      { "EbeamLTUAngXDamage",   Pds::Bld::BldDataEBeamV3::EbeamLTUAngXDamage },
      { "EbeamLTUAngYDamage",   Pds::Bld::BldDataEBeamV3::EbeamLTUAngYDamage },
      { "EbeamPkCurrBC2Damage", Pds::Bld::BldDataEBeamV3::EbeamPkCurrBC2Damage },
      { "EbeamEnergyBC2Damage", Pds::Bld::BldDataEBeamV3::EbeamEnergyBC2Damage },
      { "EbeamPkCurrBC1Damage", Pds::Bld::BldDataEBeamV3::EbeamPkCurrBC1Damage },
      { "EbeamEnergyBC1Damage", Pds::Bld::BldDataEBeamV3::EbeamEnergyBC1Damage },
      { 0, 0 }
  };
  pypdsdata::EnumType damageMaskEnum ( "DamageMask", damageMaskEnumValues );

  // methods
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV3, damageMask)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV3, ebeamCharge)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV3, ebeamL3Energy)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV3, ebeamLTUPosX)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV3, ebeamLTUPosY)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV3, ebeamLTUAngX)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV3, ebeamLTUAngY)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV3, ebeamPkCurrBC2)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV3, ebeamEnergyBC2)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV3, ebeamPkCurrBC1)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV3, ebeamEnergyBC1)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"uDamageMask",     damageMask,    0, "integer bit mask, see :py:class:`DamageMask` for individual bits meaning", 0},
    {"fEbeamCharge",    ebeamCharge,   0, "floating number, in nC", 0},
    {"fEbeamL3Energy",  ebeamL3Energy, 0, "floating number, in MeV", 0},
    {"fEbeamLTUPosX",   ebeamLTUPosX,  0, "floating number, in mm", 0},
    {"fEbeamLTUPosY",   ebeamLTUPosY,  0, "floating number, in mm", 0},
    {"fEbeamLTUAngX",   ebeamLTUAngX,  0, "floating number, in mrad", 0},
    {"fEbeamLTUAngY",   ebeamLTUAngY,  0, "floating number, in mrad", 0},
    {"fEbeamPkCurrBC2", ebeamPkCurrBC2, 0, "floating number, in Amps", 0},
    {"fEbeamEnergyBC2", ebeamEnergyBC2, 0, "floating number, beam position (related to energy), in mm", 0},
    {"fEbeamPkCurrBC1", ebeamPkCurrBC1, 0, "floating number, in Amps", 0},
    {"fEbeamEnergyBC1", ebeamEnergyBC1, 0, "floating number, beam position (related to energy), in mm", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Bld::BldDataEBeamV3 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Bld::BldDataEBeamV3::initType( PyObject* module )
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
pypdsdata::Bld::BldDataEBeamV3::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(DamageMask=" << std::showbase << std::hex << m_obj->damageMask() << std::dec
        << ", Charge=" << m_obj->ebeamCharge()
        << ", L3Energy=" << m_obj->ebeamL3Energy() << ", ...)";
  }
}
