//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FccdConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/FccdConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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

FccdConfigV1::FccdConfigV1 ( const Pds::FCCD::FccdConfigV1& data )
  : width(data.width())
  , height(data.height())
  , trimmedWidth(data.trimmedWidth())
  , trimmedHeight(data.trimmedHeight())
  , outputMode(data.outputMode())
{
}

hdf5pp::Type
FccdConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
FccdConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<FccdConfigV1>() ;
  confType.insert_native<uint32_t>( "width", offsetof(FccdConfigV1,width) );
  confType.insert_native<uint32_t>( "height", offsetof(FccdConfigV1,height) );
  confType.insert_native<uint32_t>( "trimmedWidth", offsetof(FccdConfigV1,trimmedWidth) );
  confType.insert_native<uint32_t>( "trimmedHeight", offsetof(FccdConfigV1,trimmedHeight) );
  confType.insert_native<uint16_t>( "outputMode", offsetof(FccdConfigV1,outputMode) );

  return confType ;
}

void
FccdConfigV1::store( const Pds::FCCD::FccdConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  FccdConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
