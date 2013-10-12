//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FliConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/FliConfigV1.h"

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

FliConfigV1::FliConfigV1 ( const Pds::Fli::ConfigV1& data )
  : width(data.width())
  , height(data.height())
  , orgX(data.orgX())
  , orgY(data.orgY())
  , binX(data.binX())
  , binY(data.binY())
  , exposureTime(data.exposureTime())
  , coolingTemp(data.coolingTemp())
  , gainIndex(data.gainIndex())
  , readoutSpeedIndex(data.readoutSpeedIndex())
  , exposureEventCode(data.exposureEventCode())
  , numDelayShots(data.numDelayShots())
{
}

hdf5pp::Type
FliConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
FliConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<FliConfigV1>() ;
  confType.insert_native<uint32_t>( "width", offsetof(FliConfigV1, width) );
  confType.insert_native<uint32_t>( "height", offsetof(FliConfigV1, height) );
  confType.insert_native<uint32_t>( "orgX", offsetof(FliConfigV1, orgX) );
  confType.insert_native<uint32_t>( "orgY", offsetof(FliConfigV1, orgY) );
  confType.insert_native<uint32_t>( "binX", offsetof(FliConfigV1, binX) );
  confType.insert_native<uint32_t>( "binY", offsetof(FliConfigV1, binY) );
  confType.insert_native<float>( "exposureTime", offsetof(FliConfigV1, exposureTime) );
  confType.insert_native<float>( "coolingTemp", offsetof(FliConfigV1, coolingTemp) );
  confType.insert_native<uint8_t>( "gainIndex", offsetof(FliConfigV1, gainIndex) );
  confType.insert_native<uint8_t>( "readoutSpeedIndex", offsetof(FliConfigV1, readoutSpeedIndex) );
  confType.insert_native<uint16_t>( "exposureEventCode", offsetof(FliConfigV1, exposureEventCode) );
  confType.insert_native<uint32_t>( "numDelayShots", offsetof(FliConfigV1, numDelayShots) );

  return confType ;
}

void
FliConfigV1::store( const Pds::Fli::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  FliConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
