//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DamagePolicy...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcInput/DamagePolicy.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcInput {

//----------------
// Constructors --
//----------------
DamagePolicy::DamagePolicy() :
    psana::Configurable("psana")
{
  m_storeOutOfOrderDamage = config("store-out-of-order-damage", false);
  m_storeUserEbeamDamage = config("store-user-ebeam-damage", true);
  m_storeDamagedConfig = config("store-damaged-config", false);
}

// return true if damage/type combo is good to store, false if we don't want it going in event.
// note - always returns True for undamaged types, but we do not store all undamaged types (such as Xtc).
bool DamagePolicy::eventDamagePolicy(Pds::Damage damage, enum Pds::TypeId::Type typeId)
{
  if (damage.value() == 0) return true;

  MsgLog(name(), debug,
      "eventDamagePolicy: nonzero damage=" << std::hex << damage.value() << " typeId=" << std::dec << typeId << " "
          << Pds::TypeId::name(typeId));

  bool userDamageBitSet = (damage.value() & (1 << Pds::Damage::UserDefined));
  uint32_t otherDamageBits = (damage.bits() & (~(1 << Pds::Damage::UserDefined)));
  bool userDamageByteSet = damage.userBits();

  if (not userDamageBitSet and userDamageByteSet) {
    MsgLog(name(), warning,
        "UserDefined damage bit is *not* set but user bits are present: damage=" << std::hex << damage.value()
            << " typeId=" << typeId << Pds::TypeId::name(typeId));
    return false;
  }

  bool userDamageOk = ((not userDamageBitSet and not userDamageByteSet)
      or (userDamageBitSet and (typeId == Pds::TypeId::Id_EBeam) and m_storeUserEbeamDamage));

  bool onlyOutOfOrderInOtherBits = otherDamageBits == (1 << Pds::Damage::OutOfOrder);
  if (userDamageOk) {
    if (otherDamageBits == 0) return true;
    if ((onlyOutOfOrderInOtherBits) and m_storeOutOfOrderDamage) return true;
  }
  MsgLog(name(), debug,
      "eventDamagePolicy: do not store, userDamageOk=" << userDamageOk << " only OutOfOrder in other bits="
          << onlyOutOfOrderInOtherBits << " store out of order=" << m_storeOutOfOrderDamage);

  return false;
}

bool DamagePolicy::configDamagePolicy(Pds::Damage damage)
{
  if (damage.value() == 0 or m_storeDamagedConfig) return true;
  return false;
}

} // namespace PSXtcInput
