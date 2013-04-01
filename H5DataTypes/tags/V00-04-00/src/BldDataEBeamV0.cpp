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
#include "H5DataTypes/BldDataEBeamV0.h"

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

BldDataEBeamV0::BldDataEBeamV0 ( const XtcType& xtc )
  : uDamageMask(xtc.uDamageMask)
  , fEbeamCharge(xtc.fEbeamCharge)
  , fEbeamL3Energy(xtc.fEbeamL3Energy)
  , fEbeamLTUPosX(xtc.fEbeamLTUPosX)
  , fEbeamLTUPosY(xtc.fEbeamLTUPosY)
  , fEbeamLTUAngX(xtc.fEbeamLTUAngX)
  , fEbeamLTUAngY(xtc.fEbeamLTUAngY)
{
}

BldDataEBeamV0::~BldDataEBeamV0 ()
{
}

hdf5pp::Type
BldDataEBeamV0::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataEBeamV0::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataEBeamV0>() ;
  type.insert_native<uint32_t>( "uDamageMask", offsetof(BldDataEBeamV0, uDamageMask) ) ;
  type.insert_native<double>( "fEbeamCharge", offsetof(BldDataEBeamV0, fEbeamCharge) ) ;
  type.insert_native<double>( "fEbeamL3Energy", offsetof(BldDataEBeamV0, fEbeamL3Energy) ) ;
  type.insert_native<double>( "fEbeamLTUPosX", offsetof(BldDataEBeamV0, fEbeamLTUPosX) ) ;
  type.insert_native<double>( "fEbeamLTUPosY", offsetof(BldDataEBeamV0, fEbeamLTUPosY) ) ;
  type.insert_native<double>( "fEbeamLTUAngX", offsetof(BldDataEBeamV0, fEbeamLTUAngX) ) ;
  type.insert_native<double>( "fEbeamLTUAngY", offsetof(BldDataEBeamV0, fEbeamLTUAngY) ) ;

  return type ;
}

} // namespace H5DataTypes
