//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataGMDV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataGMDV1.h"

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

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

BldDataGMDV1::BldDataGMDV1 ( const XtcType& xtc )
  : fMilliJoulesPerPulse(xtc.fMilliJoulesPerPulse)
  , fMilliJoulesAverage(xtc.fMilliJoulesAverage)
  , fCorrectedSumPerPulse(xtc.fCorrectedSumPerPulse)
  , fBgValuePerSample(xtc.fBgValuePerSample)
  , fRelativeEnergyPerPulse(xtc.fRelativeEnergyPerPulse)
  , fSpare1(xtc.fSpare1)
{
}

BldDataGMDV1::~BldDataGMDV1 ()
{
}

hdf5pp::Type
BldDataGMDV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataGMDV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataGMDV1>() ;
  type.insert_native<double>( "fMilliJoulesPerPulse", offsetof(BldDataGMDV1, fMilliJoulesPerPulse) ) ;
  type.insert_native<double>( "fMilliJoulesAverage", offsetof(BldDataGMDV1, fMilliJoulesAverage) ) ;
  type.insert_native<double>( "fCorrectedSumPerPulse", offsetof(BldDataGMDV1, fCorrectedSumPerPulse) ) ;
  type.insert_native<double>( "fBgValuePerSample", offsetof(BldDataGMDV1, fBgValuePerSample) ) ;
  type.insert_native<double>( "fRelativeEnergyPerPulse", offsetof(BldDataGMDV1, fRelativeEnergyPerPulse) ) ;
  type.insert_native<double>( "fSpare1", offsetof(BldDataGMDV1, fSpare1) ) ;

  return type ;
}

} // namespace H5DataTypes
