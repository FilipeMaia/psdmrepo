//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataPhaseCavity...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataPhaseCavity.h"

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

BldDataPhaseCavity::BldDataPhaseCavity ( const XtcType& xtc )
  : fFitTime1(xtc.fFitTime1)
  , fFitTime2(xtc.fFitTime2)
  , fCharge1(xtc.fCharge1)
  , fCharge2(xtc.fCharge2)
{
}

hdf5pp::Type
BldDataPhaseCavity::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataPhaseCavity::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataPhaseCavity>() ;
  type.insert_native<double>( "fFitTime1", offsetof(BldDataPhaseCavity, fFitTime1) ) ;
  type.insert_native<double>( "fFitTime2", offsetof(BldDataPhaseCavity, fFitTime2) ) ;
  type.insert_native<double>( "fCharge1", offsetof(BldDataPhaseCavity, fCharge1) ) ;
  type.insert_native<double>( "fCharge2", offsetof(BldDataPhaseCavity, fCharge2) ) ;

  return type ;
}


} // namespace H5DataTypes
