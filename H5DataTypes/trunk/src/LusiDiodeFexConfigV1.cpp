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
  std::copy( data.base, data.base+XtcType::NRANGES, m_data.base);
  std::copy( data.scale, data.scale+XtcType::NRANGES, m_data.scale);
}

hdf5pp::Type
LusiDiodeFexConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
LusiDiodeFexConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<LusiDiodeFexConfigV1_Data>() ;
  confType.insert_native<float>( "base", offsetof(LusiDiodeFexConfigV1_Data, base), XtcType::NRANGES );
  confType.insert_native<float>( "scale", offsetof(LusiDiodeFexConfigV1_Data, scale), XtcType::NRANGES );

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
