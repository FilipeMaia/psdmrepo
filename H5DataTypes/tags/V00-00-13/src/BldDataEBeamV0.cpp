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
#include "Lusi/Lusi.h"

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
{
  m_data.uDamageMask = xtc.uDamageMask ;
  m_data.fEbeamCharge = xtc.fEbeamCharge ;
  m_data.fEbeamL3Energy = xtc.fEbeamL3Energy ;
  m_data.fEbeamLTUPosX = xtc.fEbeamLTUPosX ;
  m_data.fEbeamLTUPosY = xtc.fEbeamLTUPosY ;
  m_data.fEbeamLTUAngX = xtc.fEbeamLTUAngX ;
  m_data.fEbeamLTUAngY = xtc.fEbeamLTUAngY ;
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
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataEBeamV0_Data>() ;
  type.insert_native<uint32_t>( "uDamageMask", offsetof(BldDataEBeamV0_Data,uDamageMask) ) ;
  type.insert_native<double>( "fEbeamCharge", offsetof(BldDataEBeamV0_Data,fEbeamCharge) ) ;
  type.insert_native<double>( "fEbeamL3Energy", offsetof(BldDataEBeamV0_Data,fEbeamL3Energy) ) ;
  type.insert_native<double>( "fEbeamLTUPosX", offsetof(BldDataEBeamV0_Data,fEbeamLTUPosX) ) ;
  type.insert_native<double>( "fEbeamLTUPosY", offsetof(BldDataEBeamV0_Data,fEbeamLTUPosY) ) ;
  type.insert_native<double>( "fEbeamLTUAngX", offsetof(BldDataEBeamV0_Data,fEbeamLTUAngX) ) ;
  type.insert_native<double>( "fEbeamLTUAngY", offsetof(BldDataEBeamV0_Data,fEbeamLTUAngY) ) ;

  return type ;
}

} // namespace H5DataTypes
