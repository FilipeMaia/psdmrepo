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
  , fHvMeshElectron(xtc.fHvMeshElectron)
  , fHvMeshIon(xtc.fHvMeshIon)
  , fHvMultIon(xtc.fHvMultIon)
  , fChargeQ(xtc.fChargeQ)
  , fPhotonEnergy(xtc.fPhotonEnergy)
  , fMultPulseIntensity(xtc.fMultPulseIntensity)
  , fKeithleyPulseIntensity(xtc.fKeithleyPulseIntensity)
  , fPulseEnergy(xtc.fPulseEnergy)
  , fPulseEnergyFEE(xtc.fPulseEnergyFEE)
  , fTransmission(xtc.fTransmission)
  , fTransmissionFEE(xtc.fTransmissionFEE)
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
  type.insert_native<double>( "fHvMeshElectron", offsetof(BldDataGMDV0, fHvMeshElectron) ) ;
  type.insert_native<double>( "fHvMeshIon", offsetof(BldDataGMDV0, fHvMeshIon) ) ;
  type.insert_native<double>( "fHvMultIon", offsetof(BldDataGMDV0, fHvMultIon) ) ;
  type.insert_native<double>( "fChargeQ", offsetof(BldDataGMDV0, fChargeQ) ) ;
  type.insert_native<double>( "fPhotonEnergy", offsetof(BldDataGMDV0, fPhotonEnergy) ) ;
  type.insert_native<double>( "fMultPulseIntensity", offsetof(BldDataGMDV0, fMultPulseIntensity) ) ;
  type.insert_native<double>( "fKeithleyPulseIntensity", offsetof(BldDataGMDV0, fKeithleyPulseIntensity) ) ;
  type.insert_native<double>( "fPulseEnergy", offsetof(BldDataGMDV0, fPulseEnergy) ) ;
  type.insert_native<double>( "fPulseEnergyFEE", offsetof(BldDataGMDV0, fPulseEnergyFEE) ) ;
  type.insert_native<double>( "fTransmission", offsetof(BldDataGMDV0, fTransmission) ) ;
  type.insert_native<double>( "fTransmissionFEE", offsetof(BldDataGMDV0, fTransmissionFEE) ) ;
  type.insert_native<double>( "fSpare6", offsetof(BldDataGMDV0, fSpare6) ) ;

  return type ;
}

} // namespace H5DataTypes
