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
      { "EbeamChargeDamage",    Pds::BldDataEBeamV0::EbeamChargeDamage },
      { "EbeamL3EnergyDamage",  Pds::BldDataEBeamV0::EbeamL3EnergyDamage },
      { "EbeamLTUPosXDamage",   Pds::BldDataEBeamV0::EbeamLTUPosXDamage },
      { "EbeamLTUPosYDamage",   Pds::BldDataEBeamV0::EbeamLTUPosYDamage },
      { "EbeamLTUAngXDamage",   Pds::BldDataEBeamV0::EbeamLTUAngXDamage },
      { "EbeamLTUAngYDamage",   Pds::BldDataEBeamV0::EbeamLTUAngYDamage },
      { 0, 0 }
  };
  pypdsdata::EnumType damageMaskEnum ( "DamageMask", damageMaskEnumValues );

  // methods
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, uDamageMask)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamCharge)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamL3Energy)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamLTUPosX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamLTUPosY)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamLTUAngX)
  MEMBER_WRAPPER(pypdsdata::BldDataEBeamV0, fEbeamLTUAngY)

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

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "DamageMask", damageMaskEnum.type() );

  BaseType::initType( "BldDataEBeamV0", module );
}

void
pypdsdata::BldDataEBeamV0::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(DamageMask=" << std::showbase << std::hex << m_obj->uDamageMask << std::dec
        << ", Charge=" << m_obj->fEbeamCharge
        << ", L3Energy=" << m_obj->fEbeamL3Energy << ", ...)";
  }
}
