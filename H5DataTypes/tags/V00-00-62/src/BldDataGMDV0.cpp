//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataGMDV0...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataGMDV0.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  hdf5pp::Type _strType( size_t size )
  {
    hdf5pp::Type strType = hdf5pp::Type::Copy(H5T_C_S1);
    strType.set_size( size ) ;
    return strType ;
  }


  hdf5pp::Type _strGasType()
  {
    static hdf5pp::Type strType = _strType( 32 );
    return strType ;
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

BldDataGMDV0::BldDataGMDV0 ( const XtcType& xtc )
  : fPressure(xtc.fPressure)
  , fTemperature(xtc.fTemperature)
  , fCurrent(xtc.fCurrent)
  , iHvMeshElectron(xtc.iHvMeshElectron)
  , iHvMeshIon(xtc.iHvMeshIon)
  , iHvMultIon(xtc.iHvMultIon)
  , fChargeQ(xtc.fChargeQ)
  , fPhotonEnergy(xtc.fPhotonEnergy)
  , fPhotonsPerPulse(xtc.fPhotonsPerPulse)
  , fSpare1(xtc.fSpare1)
  , fSpare2(xtc.fSpare2)
  , fSpare3(xtc.fSpare3)
  , fSpare4(xtc.fSpare4)
  , fSpare5(xtc.fSpare5)
  , fSpare6(xtc.fSpare6)
{
  std::copy(xtc.strGasType+0, xtc.strGasType+32, strGasType);
}

BldDataGMDV0::~BldDataGMDV0 ()
{
}

hdf5pp::Type
BldDataGMDV0::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataGMDV0::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataGMDV0>() ;
  type.insert( "strGasType", offsetof(BldDataGMDV0, strGasType), _strGasType() ) ;
  type.insert_native<double>( "fPressure", offsetof(BldDataGMDV0, fPressure) ) ;
  type.insert_native<double>( "fTemperature", offsetof(BldDataGMDV0, fTemperature) ) ;
  type.insert_native<double>( "fCurrent", offsetof(BldDataGMDV0, fCurrent) ) ;
  type.insert_native<int32_t>( "iHvMeshElectron", offsetof(BldDataGMDV0, iHvMeshElectron) ) ;
  type.insert_native<int32_t>( "iHvMeshIon", offsetof(BldDataGMDV0, iHvMeshIon) ) ;
  type.insert_native<int32_t>( "iHvMultIon", offsetof(BldDataGMDV0, iHvMultIon) ) ;
  type.insert_native<double>( "fChargeQ", offsetof(BldDataGMDV0, fChargeQ) ) ;
  type.insert_native<double>( "fPhotonEnergy", offsetof(BldDataGMDV0, fPhotonEnergy) ) ;
  type.insert_native<double>( "fPhotonsPerPulse", offsetof(BldDataGMDV0, fPhotonsPerPulse) ) ;
  type.insert_native<double>( "fSpare1", offsetof(BldDataGMDV0, fSpare1) ) ;
  type.insert_native<double>( "fSpare2", offsetof(BldDataGMDV0, fSpare2) ) ;
  type.insert_native<double>( "fSpare3", offsetof(BldDataGMDV0, fSpare3) ) ;
  type.insert_native<double>( "fSpare4", offsetof(BldDataGMDV0, fSpare4) ) ;
  type.insert_native<double>( "fSpare5", offsetof(BldDataGMDV0, fSpare5) ) ;
  type.insert_native<double>( "fSpare6", offsetof(BldDataGMDV0, fSpare6) ) ;

  return type ;
}

} // namespace H5DataTypes
