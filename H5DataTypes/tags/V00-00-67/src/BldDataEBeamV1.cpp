//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataEBeamV1.h"

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

BldDataEBeamV1::BldDataEBeamV1 ( const XtcType& xtc )
  : uDamageMask(xtc.uDamageMask)
  , fEbeamCharge(xtc.fEbeamCharge)
  , fEbeamL3Energy(xtc.fEbeamL3Energy)
  , fEbeamLTUPosX(xtc.fEbeamLTUPosX)
  , fEbeamLTUPosY(xtc.fEbeamLTUPosY)
  , fEbeamLTUAngX(xtc.fEbeamLTUAngX)
  , fEbeamLTUAngY(xtc.fEbeamLTUAngY)
  , fEbeamPkCurrBC2(xtc.fEbeamPkCurrBC2)
{
}

BldDataEBeamV1::~BldDataEBeamV1 ()
{
}

hdf5pp::Type
BldDataEBeamV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataEBeamV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataEBeamV1>() ;
  type.insert_native<uint32_t>( "uDamageMask", offsetof(BldDataEBeamV1, uDamageMask) ) ;
  type.insert_native<double>( "fEbeamCharge", offsetof(BldDataEBeamV1, fEbeamCharge) ) ;
  type.insert_native<double>( "fEbeamL3Energy", offsetof(BldDataEBeamV1, fEbeamL3Energy) ) ;
  type.insert_native<double>( "fEbeamLTUPosX", offsetof(BldDataEBeamV1, fEbeamLTUPosX) ) ;
  type.insert_native<double>( "fEbeamLTUPosY", offsetof(BldDataEBeamV1, fEbeamLTUPosY) ) ;
  type.insert_native<double>( "fEbeamLTUAngX", offsetof(BldDataEBeamV1, fEbeamLTUAngX) ) ;
  type.insert_native<double>( "fEbeamLTUAngY", offsetof(BldDataEBeamV1, fEbeamLTUAngY) ) ;
  type.insert_native<double>( "fEbeamPkCurrBC2", offsetof(BldDataEBeamV1, fEbeamPkCurrBC2) ) ;

  return type ;
}

} // namespace H5DataTypes
