#ifndef PSXTCINPUT_DAMAGEPOLICY_H
#define PSXTCINPUT_DAMAGEPOLICY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DamagePolicy.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psana/Configurable.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Damage.hh"
#include "pdsdata/xtc/TypeId.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcInput {

/// @addtogroup PSXtcInput

/**
 *  @ingroup PSXtcInput
 *
 *  @brief Class which determines where data with particular damage needs to be stored 
 *  in psana event.
 *
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class DamagePolicy : public psana::Configurable {
public:

  // Default constructor
  DamagePolicy();

  /// determines if an xtc type should be stored in the event
  bool eventDamagePolicy(Pds::Damage damage, enum Pds::TypeId::Type typeId);

  /// determines if an xtc type should be stored in the config
  bool configDamagePolicy(Pds::Damage damage);

protected:

private:

  bool m_storeOutOfOrderDamage;                       ///< if false, do not parse Xtc Type, just report damage
  bool m_storeUserEbeamDamage;                        ///< if true, make exception for user damage if for Ebeam
  bool m_storeDamagedConfig;                          ///< if true, store damaged config

};

} // namespace PSXtcInput

#endif // PSXTCINPUT_DAMAGEPOLICY_H
