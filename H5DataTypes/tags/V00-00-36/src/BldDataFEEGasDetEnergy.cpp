//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataFEEGasDetEnergy...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataFEEGasDetEnergy.h"

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

BldDataFEEGasDetEnergy::BldDataFEEGasDetEnergy ( const XtcType& xtc )
{
  m_data.f_11_ENRC = xtc.f_11_ENRC ;
  m_data.f_12_ENRC = xtc.f_12_ENRC ;
  m_data.f_21_ENRC = xtc.f_21_ENRC ;
  m_data.f_22_ENRC = xtc.f_22_ENRC ;
}

BldDataFEEGasDetEnergy::~BldDataFEEGasDetEnergy ()
{
}

hdf5pp::Type
BldDataFEEGasDetEnergy::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataFEEGasDetEnergy::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataFEEGasDetEnergy_Data>() ;
  type.insert_native<double>( "f_11_ENRC", offsetof(BldDataFEEGasDetEnergy_Data,f_11_ENRC) ) ;
  type.insert_native<double>( "f_12_ENRC", offsetof(BldDataFEEGasDetEnergy_Data,f_12_ENRC) ) ;
  type.insert_native<double>( "f_21_ENRC", offsetof(BldDataFEEGasDetEnergy_Data,f_21_ENRC) ) ;
  type.insert_native<double>( "f_22_ENRC", offsetof(BldDataFEEGasDetEnergy_Data,f_22_ENRC) ) ;

  return type ;
}

} // namespace H5DataTypes
