//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeam...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataEBeam.h"

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

BldDataEBeam::BldDataEBeam ( const XtcType& xtc )
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

BldDataEBeam::~BldDataEBeam ()
{
}

hdf5pp::Type
BldDataEBeam::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataEBeam::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataEBeam_Data>() ;
  type.insert_native<uint32_t>( "uDamageMask", offsetof(BldDataEBeam_Data,uDamageMask) ) ;
  type.insert_native<double>( "fEbeamCharge", offsetof(BldDataEBeam_Data,fEbeamCharge) ) ;
  type.insert_native<double>( "fEbeamL3Energy", offsetof(BldDataEBeam_Data,fEbeamL3Energy) ) ;
  type.insert_native<double>( "fEbeamLTUPosX", offsetof(BldDataEBeam_Data,fEbeamLTUPosX) ) ;
  type.insert_native<double>( "fEbeamLTUPosY", offsetof(BldDataEBeam_Data,fEbeamLTUPosY) ) ;
  type.insert_native<double>( "fEbeamLTUAngX", offsetof(BldDataEBeam_Data,fEbeamLTUAngX) ) ;
  type.insert_native<double>( "fEbeamLTUAngY", offsetof(BldDataEBeam_Data,fEbeamLTUAngY) ) ;
  type.insert_native<double>( "fEbeamPkCurrBC2", offsetof(BldDataEBeam_Data,fEbeamPkCurrBC2) ) ;

  return type ;
}

} // namespace H5DataTypes
