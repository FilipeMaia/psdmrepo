//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiDiodeFexConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/LusiDiodeFexConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

LusiDiodeFexConfigV1::LusiDiodeFexConfigV1 ( const XtcType& data )
{
  const ndarray<const float, 1>& ndbase = data.base();
  std::copy( ndbase.begin(), ndbase.end(), base);
  const ndarray<const float, 1>& ndscale = data.scale();
  std::copy( ndscale.begin(), ndscale.end(), scale);
}

hdf5pp::Type
LusiDiodeFexConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
LusiDiodeFexConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<LusiDiodeFexConfigV1>() ;
  confType.insert_native<float>( "base", offsetof(LusiDiodeFexConfigV1, base), XtcType::NRANGES );
  confType.insert_native<float>( "scale", offsetof(LusiDiodeFexConfigV1, scale), XtcType::NRANGES );

  return confType ;
}

void
LusiDiodeFexConfigV1::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  LusiDiodeFexConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
