//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcDamage...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/XtcDamage.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

//----------------
// Constructors --
//----------------
XtcDamage::XtcDamage (Pds::Damage damage)
  : bits(damage.bits())
  , DroppedContribution((damage.bits() & (1 << Pds::Damage::DroppedContribution)) != 0)
  , OutOfOrder((damage.bits() & (1 << Pds::Damage::OutOfOrder)) != 0)
  , OutOfSynch((damage.bits() & (1 << Pds::Damage::OutOfSynch)) != 0)
  , UserDefined((damage.bits() & (1 << Pds::Damage::UserDefined)) != 0)
  , IncompleteContribution((damage.bits() & (1 << Pds::Damage::IncompleteContribution)) != 0)
  , userBits(damage.userBits())
{
}

hdf5pp::Type
XtcDamage::stored_type()
{
  return native_type();
}

hdf5pp::Type
XtcDamage::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType(sizeof(XtcDamage));
  type.insert_native<uint32_t>("bits", offsetof(XtcDamage, bits));
  type.insert_native<uint8_t>("DroppedContribution", offsetof(XtcDamage, DroppedContribution));
  type.insert_native<uint8_t>("OutOfOrder", offsetof(XtcDamage, OutOfOrder));
  type.insert_native<uint8_t>("OutOfSynch", offsetof(XtcDamage, OutOfSynch));
  type.insert_native<uint8_t>("UserDefined", offsetof(XtcDamage, UserDefined));
  type.insert_native<uint8_t>("IncompleteContribution", offsetof(XtcDamage, IncompleteContribution));
  type.insert_native<uint8_t>("userBits", offsetof(XtcDamage, userBits));
  return type ;
}

} // namespace H5DataTypes
