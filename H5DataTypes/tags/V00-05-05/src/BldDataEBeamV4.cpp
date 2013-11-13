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
#include "H5DataTypes/BldDataEBeamV4.h"

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

BldDataEBeamV4::BldDataEBeamV4 ( const XtcType& xtc )
  : uDamageMask(xtc.damageMask())
  , fEbeamCharge(xtc.ebeamCharge())
  , fEbeamL3Energy(xtc.ebeamL3Energy())
  , fEbeamLTUPosX(xtc.ebeamLTUPosX())
  , fEbeamLTUPosY(xtc.ebeamLTUPosY())
  , fEbeamLTUAngX(xtc.ebeamLTUAngX())
  , fEbeamLTUAngY(xtc.ebeamLTUAngY())
  , fEbeamPkCurrBC2(xtc.ebeamPkCurrBC2())
  , fEbeamEnergyBC2(xtc.ebeamEnergyBC2())
  , fEbeamPkCurrBC1(xtc.ebeamPkCurrBC1())
  , fEbeamEnergyBC1(xtc.ebeamEnergyBC1())
  , fEbeamUndPosX(xtc.ebeamUndPosX())
  , fEbeamUndPosY(xtc.ebeamUndPosY())
  , fEbeamUndAngX(xtc.ebeamUndAngX())
  , fEbeamUndAngY(xtc.ebeamUndAngY())
{
}

BldDataEBeamV4::~BldDataEBeamV4 ()
{
}

hdf5pp::Type
BldDataEBeamV4::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataEBeamV4::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataEBeamV4>() ;
  type.insert_native<uint32_t>( "uDamageMask", offsetof(BldDataEBeamV4, uDamageMask) ) ;
  type.insert_native<double>( "fEbeamCharge", offsetof(BldDataEBeamV4, fEbeamCharge) ) ;
  type.insert_native<double>( "fEbeamL3Energy", offsetof(BldDataEBeamV4, fEbeamL3Energy) ) ;
  type.insert_native<double>( "fEbeamLTUPosX", offsetof(BldDataEBeamV4, fEbeamLTUPosX) ) ;
  type.insert_native<double>( "fEbeamLTUPosY", offsetof(BldDataEBeamV4, fEbeamLTUPosY) ) ;
  type.insert_native<double>( "fEbeamLTUAngX", offsetof(BldDataEBeamV4, fEbeamLTUAngX) ) ;
  type.insert_native<double>( "fEbeamLTUAngY", offsetof(BldDataEBeamV4, fEbeamLTUAngY) ) ;
  type.insert_native<double>( "fEbeamPkCurrBC2", offsetof(BldDataEBeamV4, fEbeamPkCurrBC2) ) ;
  type.insert_native<double>( "fEbeamEnergyBC2", offsetof(BldDataEBeamV4, fEbeamEnergyBC2) ) ;
  type.insert_native<double>( "fEbeamPkCurrBC1", offsetof(BldDataEBeamV4, fEbeamPkCurrBC1) ) ;
  type.insert_native<double>( "fEbeamEnergyBC1", offsetof(BldDataEBeamV4, fEbeamEnergyBC1) ) ;
  type.insert_native<double>( "fEbeamUndPosX", offsetof(BldDataEBeamV4, fEbeamUndPosX) ) ;
  type.insert_native<double>( "fEbeamUndPosY", offsetof(BldDataEBeamV4, fEbeamUndPosY) ) ;
  type.insert_native<double>( "fEbeamUndAngX", offsetof(BldDataEBeamV4, fEbeamUndAngX) ) ;
  type.insert_native<double>( "fEbeamUndAngY", offsetof(BldDataEBeamV4, fEbeamUndAngY) ) ;

  return type ;
}

} // namespace H5DataTypes
