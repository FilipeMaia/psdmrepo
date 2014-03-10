//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataSpectrometerV0...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataSpectrometerV0.h"

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

//----------------
// Constructors --
//----------------
BldDataSpectrometerV0::BldDataSpectrometerV0 (const XtcType& xtc)
{
  const ndarray<const uint32_t, 1>& hproj = xtc.hproj();
  std::copy(hproj.begin(), hproj.end(), this->hproj);
  const ndarray<const uint32_t, 1>& vproj = xtc.vproj();
  std::copy(vproj.begin(), vproj.end(), this->vproj);
}

hdf5pp::Type
BldDataSpectrometerV0::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataSpectrometerV0::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataSpectrometerV0>() ;
  type.insert_native<uint32_t>( "hproj", offsetof(BldDataSpectrometerV0, hproj), 1024 ) ;
  type.insert_native<uint32_t>( "vproj", offsetof(BldDataSpectrometerV0, vproj), 256 ) ;

  return type ;
}

} // namespace H5DataTypes
