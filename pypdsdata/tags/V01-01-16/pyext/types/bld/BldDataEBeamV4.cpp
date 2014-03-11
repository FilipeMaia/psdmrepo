//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV4...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataEBeamV4.h"

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
      { "EbeamChargeDamage",    Pds::Bld::BldDataEBeamV4::EbeamChargeDamage },
      { "EbeamL3EnergyDamage",  Pds::Bld::BldDataEBeamV4::EbeamL3EnergyDamage },
      { "EbeamLTUPosXDamage",   Pds::Bld::BldDataEBeamV4::EbeamLTUPosXDamage },
      { "EbeamLTUPosYDamage",   Pds::Bld::BldDataEBeamV4::EbeamLTUPosYDamage },
      { "EbeamLTUAngXDamage",   Pds::Bld::BldDataEBeamV4::EbeamLTUAngXDamage },
      { "EbeamLTUAngYDamage",   Pds::Bld::BldDataEBeamV4::EbeamLTUAngYDamage },
      { "EbeamPkCurrBC2Damage", Pds::Bld::BldDataEBeamV4::EbeamPkCurrBC2Damage },
      { "EbeamEnergyBC2Damage", Pds::Bld::BldDataEBeamV4::EbeamEnergyBC2Damage },
      { "EbeamPkCurrBC1Damage", Pds::Bld::BldDataEBeamV4::EbeamPkCurrBC1Damage },
      { "EbeamEnergyBC1Damage", Pds::Bld::BldDataEBeamV4::EbeamEnergyBC1Damage },
      { "EbeamUndPosXDamage",   Pds::Bld::BldDataEBeamV4::EbeamUndPosXDamage },
      { "EbeamUndPosYDamage",   Pds::Bld::BldDataEBeamV4::EbeamUndPosYDamage },
      { "EbeamUndAngXDamage",   Pds::Bld::BldDataEBeamV4::EbeamUndAngXDamage },
      { "EbeamUndAngYDamage",   Pds::Bld::BldDataEBeamV4::EbeamUndAngYDamage },
      { 0, 0 }
  };
  pypdsdata::EnumType damageMaskEnum ( "DamageMask", damageMaskEnumValues );

  // methods
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, damageMask)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamCharge)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamL3Energy)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamLTUPosX)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamLTUPosY)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamLTUAngX)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamLTUAngY)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamPkCurrBC2)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamEnergyBC2)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamPkCurrBC1)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamEnergyBC1)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamUndPosX)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamUndPosY)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamUndAngX)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV4, ebeamUndAngY)

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
    {"fEbeamUndPosX",   ebeamUndPosX,   0, "Undulator launch feedback (BPMs U4 through U10) beam x-position in mm.", 0},
    {"fEbeamUndPosY",   ebeamUndPosY,   0, "Undulator launch feedback beam y-position in mm.", 0},
    {"fEbeamUndAngX",   ebeamUndAngX,   0, "Undulator launch feedback beam x-angle in mrad.", 0},
    {"fEbeamUndAngY",   ebeamUndAngY,   0, "Undulator launch feedback beam y-angle in mrad.", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Bld::BldDataEBeamV4 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Bld::BldDataEBeamV4::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "DamageMask", damageMaskEnum.type() );

  BaseType::initType( "BldDataEBeamV4", module );
}

void
pypdsdata::Bld::BldDataEBeamV4::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(DamageMask=" << std::showbase << std::hex << m_obj->damageMask() << std::dec
        << ", Charge=" << m_obj->ebeamCharge()
        << ", L3Energy=" << m_obj->ebeamL3Energy() << ", ...)";
  }
}
