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
#include "H5DataTypes/BldDataEBeamV5.h"

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

BldDataEBeamV5::BldDataEBeamV5 ( const XtcType& xtc )
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
  , fEbeamXTCAVAmpl(xtc.ebeamXTCAVAmpl())
  , fEbeamXTCAVPhase(xtc.ebeamXTCAVPhase())
  , fEbeamDumpCharge(xtc.ebeamDumpCharge())
{
}

BldDataEBeamV5::~BldDataEBeamV5 ()
{
}

hdf5pp::Type
BldDataEBeamV5::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataEBeamV5::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataEBeamV5>() ;
  type.insert_native<uint32_t>( "uDamageMask", offsetof(BldDataEBeamV5, uDamageMask) ) ;
  type.insert_native<double>( "fEbeamCharge", offsetof(BldDataEBeamV5, fEbeamCharge) ) ;
  type.insert_native<double>( "fEbeamL3Energy", offsetof(BldDataEBeamV5, fEbeamL3Energy) ) ;
  type.insert_native<double>( "fEbeamLTUPosX", offsetof(BldDataEBeamV5, fEbeamLTUPosX) ) ;
  type.insert_native<double>( "fEbeamLTUPosY", offsetof(BldDataEBeamV5, fEbeamLTUPosY) ) ;
  type.insert_native<double>( "fEbeamLTUAngX", offsetof(BldDataEBeamV5, fEbeamLTUAngX) ) ;
  type.insert_native<double>( "fEbeamLTUAngY", offsetof(BldDataEBeamV5, fEbeamLTUAngY) ) ;
  type.insert_native<double>( "fEbeamPkCurrBC2", offsetof(BldDataEBeamV5, fEbeamPkCurrBC2) ) ;
  type.insert_native<double>( "fEbeamEnergyBC2", offsetof(BldDataEBeamV5, fEbeamEnergyBC2) ) ;
  type.insert_native<double>( "fEbeamPkCurrBC1", offsetof(BldDataEBeamV5, fEbeamPkCurrBC1) ) ;
  type.insert_native<double>( "fEbeamEnergyBC1", offsetof(BldDataEBeamV5, fEbeamEnergyBC1) ) ;
  type.insert_native<double>( "fEbeamUndPosX", offsetof(BldDataEBeamV5, fEbeamUndPosX) ) ;
  type.insert_native<double>( "fEbeamUndPosY", offsetof(BldDataEBeamV5, fEbeamUndPosY) ) ;
  type.insert_native<double>( "fEbeamUndAngX", offsetof(BldDataEBeamV5, fEbeamUndAngX) ) ;
  type.insert_native<double>( "fEbeamUndAngY", offsetof(BldDataEBeamV5, fEbeamUndAngY) ) ;
  type.insert_native<double>( "fEbeamXTCAVAmpl", offsetof(BldDataEBeamV5, fEbeamXTCAVAmpl) );
  type.insert_native<double>( "fEbeamXTCAVPhase", offsetof(BldDataEBeamV5, fEbeamXTCAVPhase) );
  type.insert_native<double>( "fEbeamDumpCharge", offsetof(BldDataEBeamV5, fEbeamDumpCharge) );

  return type ;
}

} // namespace H5DataTypes
