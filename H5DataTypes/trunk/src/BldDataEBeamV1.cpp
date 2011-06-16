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
{
  m_data.uDamageMask = xtc.uDamageMask ;
  m_data.fEbeamCharge = xtc.fEbeamCharge ;
  m_data.fEbeamL3Energy = xtc.fEbeamL3Energy ;
  m_data.fEbeamLTUPosX = xtc.fEbeamLTUPosX ;
  m_data.fEbeamLTUPosY = xtc.fEbeamLTUPosY ;
  m_data.fEbeamLTUAngX = xtc.fEbeamLTUAngX ;
  m_data.fEbeamLTUAngY = xtc.fEbeamLTUAngY ;
  m_data.fEbeamPkCurrBC2 = xtc.fEbeamPkCurrBC2 ;
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
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataEBeamV1_Data>() ;
  type.insert_native<uint32_t>( "uDamageMask", offsetof(BldDataEBeamV1_Data,uDamageMask) ) ;
  type.insert_native<double>( "fEbeamCharge", offsetof(BldDataEBeamV1_Data,fEbeamCharge) ) ;
  type.insert_native<double>( "fEbeamL3Energy", offsetof(BldDataEBeamV1_Data,fEbeamL3Energy) ) ;
  type.insert_native<double>( "fEbeamLTUPosX", offsetof(BldDataEBeamV1_Data,fEbeamLTUPosX) ) ;
  type.insert_native<double>( "fEbeamLTUPosY", offsetof(BldDataEBeamV1_Data,fEbeamLTUPosY) ) ;
  type.insert_native<double>( "fEbeamLTUAngX", offsetof(BldDataEBeamV1_Data,fEbeamLTUAngX) ) ;
  type.insert_native<double>( "fEbeamLTUAngY", offsetof(BldDataEBeamV1_Data,fEbeamLTUAngY) ) ;
  type.insert_native<double>( "fEbeamPkCurrBC2", offsetof(BldDataEBeamV1_Data,fEbeamPkCurrBC2) ) ;

  return type ;
}

} // namespace H5DataTypes
