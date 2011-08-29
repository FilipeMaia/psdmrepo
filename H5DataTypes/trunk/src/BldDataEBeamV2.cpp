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
#include "H5DataTypes/BldDataEBeamV2.h"

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

BldDataEBeamV2::BldDataEBeamV2 ( const XtcType& xtc )
  : uDamageMask(xtc.uDamageMask)
  , fEbeamCharge(xtc.fEbeamCharge)
  , fEbeamL3Energy(xtc.fEbeamL3Energy)
  , fEbeamLTUPosX(xtc.fEbeamLTUPosX)
  , fEbeamLTUPosY(xtc.fEbeamLTUPosY)
  , fEbeamLTUAngX(xtc.fEbeamLTUAngX)
  , fEbeamLTUAngY(xtc.fEbeamLTUAngY)
  , fEbeamPkCurrBC2(xtc.fEbeamPkCurrBC2)
  , fEbeamEnergyBC2(xtc.fEbeamEnergyBC2)
{
}

BldDataEBeamV2::~BldDataEBeamV2 ()
{
}

hdf5pp::Type
BldDataEBeamV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataEBeamV2::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataEBeamV2>() ;
  type.insert_native<uint32_t>( "uDamageMask", offsetof(BldDataEBeamV2, uDamageMask) ) ;
  type.insert_native<double>( "fEbeamCharge", offsetof(BldDataEBeamV2, fEbeamCharge) ) ;
  type.insert_native<double>( "fEbeamL3Energy", offsetof(BldDataEBeamV2, fEbeamL3Energy) ) ;
  type.insert_native<double>( "fEbeamLTUPosX", offsetof(BldDataEBeamV2, fEbeamLTUPosX) ) ;
  type.insert_native<double>( "fEbeamLTUPosY", offsetof(BldDataEBeamV2, fEbeamLTUPosY) ) ;
  type.insert_native<double>( "fEbeamLTUAngX", offsetof(BldDataEBeamV2, fEbeamLTUAngX) ) ;
  type.insert_native<double>( "fEbeamLTUAngY", offsetof(BldDataEBeamV2, fEbeamLTUAngY) ) ;
  type.insert_native<double>( "fEbeamPkCurrBC2", offsetof(BldDataEBeamV2, fEbeamPkCurrBC2) ) ;
  type.insert_native<double>( "fEbeamEnergyBC2", offsetof(BldDataEBeamV2, fEbeamEnergyBC2) ) ;

  return type ;
}

} // namespace H5DataTypes
