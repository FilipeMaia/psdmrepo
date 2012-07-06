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
#include "H5DataTypes/BldDataEBeamV3.h"

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

BldDataEBeamV3::BldDataEBeamV3 ( const XtcType& xtc )
  : uDamageMask(xtc.uDamageMask)
  , fEbeamCharge(xtc.fEbeamCharge)
  , fEbeamL3Energy(xtc.fEbeamL3Energy)
  , fEbeamLTUPosX(xtc.fEbeamLTUPosX)
  , fEbeamLTUPosY(xtc.fEbeamLTUPosY)
  , fEbeamLTUAngX(xtc.fEbeamLTUAngX)
  , fEbeamLTUAngY(xtc.fEbeamLTUAngY)
  , fEbeamPkCurrBC2(xtc.fEbeamPkCurrBC2)
  , fEbeamEnergyBC2(xtc.fEbeamEnergyBC2)
  , fEbeamPkCurrBC1(xtc.fEbeamPkCurrBC1)
  , fEbeamEnergyBC1(xtc.fEbeamEnergyBC1)
{
}

BldDataEBeamV3::~BldDataEBeamV3 ()
{
}

hdf5pp::Type
BldDataEBeamV3::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataEBeamV3::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataEBeamV3>() ;
  type.insert_native<uint32_t>( "uDamageMask", offsetof(BldDataEBeamV3, uDamageMask) ) ;
  type.insert_native<double>( "fEbeamCharge", offsetof(BldDataEBeamV3, fEbeamCharge) ) ;
  type.insert_native<double>( "fEbeamL3Energy", offsetof(BldDataEBeamV3, fEbeamL3Energy) ) ;
  type.insert_native<double>( "fEbeamLTUPosX", offsetof(BldDataEBeamV3, fEbeamLTUPosX) ) ;
  type.insert_native<double>( "fEbeamLTUPosY", offsetof(BldDataEBeamV3, fEbeamLTUPosY) ) ;
  type.insert_native<double>( "fEbeamLTUAngX", offsetof(BldDataEBeamV3, fEbeamLTUAngX) ) ;
  type.insert_native<double>( "fEbeamLTUAngY", offsetof(BldDataEBeamV3, fEbeamLTUAngY) ) ;
  type.insert_native<double>( "fEbeamPkCurrBC2", offsetof(BldDataEBeamV3, fEbeamPkCurrBC2) ) ;
  type.insert_native<double>( "fEbeamEnergyBC2", offsetof(BldDataEBeamV3, fEbeamEnergyBC2) ) ;
  type.insert_native<double>( "fEbeamPkCurrBC1", offsetof(BldDataEBeamV3, fEbeamPkCurrBC1) ) ;
  type.insert_native<double>( "fEbeamEnergyBC1", offsetof(BldDataEBeamV3, fEbeamEnergyBC1) ) ;

  return type ;
}

} // namespace H5DataTypes
