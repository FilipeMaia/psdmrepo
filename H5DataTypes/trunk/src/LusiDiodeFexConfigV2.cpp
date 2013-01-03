//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LusiDiodeFexConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/LusiDiodeFexConfigV2.h"

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

LusiDiodeFexConfigV2::LusiDiodeFexConfigV2 ( const XtcType& data )
{
  std::copy( data.base, data.base+XtcType::NRANGES, m_data.base);
  std::copy( data.scale, data.scale+XtcType::NRANGES, m_data.scale);
}

hdf5pp::Type
LusiDiodeFexConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
LusiDiodeFexConfigV2::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<LusiDiodeFexConfigV2_Data>() ;
  confType.insert_native<float>( "base", offsetof(LusiDiodeFexConfigV2_Data, base), XtcType::NRANGES );
  confType.insert_native<float>( "scale", offsetof(LusiDiodeFexConfigV2_Data, scale), XtcType::NRANGES );

  return confType ;
}

void
LusiDiodeFexConfigV2::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  LusiDiodeFexConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
