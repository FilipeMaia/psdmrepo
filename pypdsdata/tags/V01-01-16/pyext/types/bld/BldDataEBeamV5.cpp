//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV5...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataEBeamV5.h"

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
      { "EbeamChargeDamage",    Pds::Bld::BldDataEBeamV5::EbeamChargeDamage },
      { "EbeamL3EnergyDamage",  Pds::Bld::BldDataEBeamV5::EbeamL3EnergyDamage },
      { "EbeamLTUPosXDamage",   Pds::Bld::BldDataEBeamV5::EbeamLTUPosXDamage },
      { "EbeamLTUPosYDamage",   Pds::Bld::BldDataEBeamV5::EbeamLTUPosYDamage },
      { "EbeamLTUAngXDamage",   Pds::Bld::BldDataEBeamV5::EbeamLTUAngXDamage },
      { "EbeamLTUAngYDamage",   Pds::Bld::BldDataEBeamV5::EbeamLTUAngYDamage },
      { "EbeamPkCurrBC2Damage", Pds::Bld::BldDataEBeamV5::EbeamPkCurrBC2Damage },
      { "EbeamEnergyBC2Damage", Pds::Bld::BldDataEBeamV5::EbeamEnergyBC2Damage },
      { "EbeamPkCurrBC1Damage", Pds::Bld::BldDataEBeamV5::EbeamPkCurrBC1Damage },
      { "EbeamEnergyBC1Damage", Pds::Bld::BldDataEBeamV5::EbeamEnergyBC1Damage },
      { "EbeamUndPosXDamage",   Pds::Bld::BldDataEBeamV5::EbeamUndPosXDamage },
      { "EbeamUndPosYDamage",   Pds::Bld::BldDataEBeamV5::EbeamUndPosYDamage },
      { "EbeamUndAngXDamage",   Pds::Bld::BldDataEBeamV5::EbeamUndAngXDamage },
      { "EbeamUndAngYDamage",   Pds::Bld::BldDataEBeamV5::EbeamUndAngYDamage },
      { "EbeamXTCAVAmplDamage", Pds::Bld::BldDataEBeamV5::EbeamXTCAVAmplDamage },
      { "EbeamXTCAVPhaseDamage",Pds::Bld::BldDataEBeamV5::EbeamXTCAVPhaseDamage },
      { "EbeamDumpChargeDamage",Pds::Bld::BldDataEBeamV5::EbeamDumpChargeDamage },
      { 0, 0 }
  };
  pypdsdata::EnumType damageMaskEnum ( "DamageMask", damageMaskEnumValues );

  // methods
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, damageMask)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamCharge)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamL3Energy)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamLTUPosX)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamLTUPosY)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamLTUAngX)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamLTUAngY)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamPkCurrBC2)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamEnergyBC2)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamPkCurrBC1)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamEnergyBC1)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamUndPosX)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamUndPosY)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamUndAngX)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamUndAngY)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamXTCAVAmpl)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamXTCAVPhase)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataEBeamV5, ebeamDumpCharge)

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
    {"fEbeamXTCAVAmpl", ebeamXTCAVAmpl, 0, "XTCAV Amplitude in MVolt.",0},
    {"fEbeamXTCAVPhase", ebeamXTCAVPhase, 0, "XTCAV Phase in degrees.",0},
    {"fEbeamDumpCharge", ebeamDumpCharge, 0, "Bunch charge at Dump in num. electrons",0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Bld::BldDataEBeamV5 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Bld::BldDataEBeamV5::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "DamageMask", damageMaskEnum.type() );

  BaseType::initType( "BldDataEBeamV5", module );
}

void
pypdsdata::Bld::BldDataEBeamV5::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(DamageMask=" << std::showbase << std::hex << m_obj->damageMask() << std::dec
        << ", Charge=" << m_obj->ebeamCharge()
        << ", L3Energy=" << m_obj->ebeamL3Energy() << ", ...)";
  }
}
