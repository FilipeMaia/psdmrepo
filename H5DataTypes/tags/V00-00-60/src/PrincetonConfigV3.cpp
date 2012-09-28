//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonConfigV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PrincetonConfigV3.h"

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

PrincetonConfigV3::PrincetonConfigV3 ( const Pds::Princeton::ConfigV3& data )
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
PrincetonConfigV3::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PrincetonConfigV3::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<PrincetonConfigV3>() ;
  confType.insert_native<uint32_t>( "width", offsetof(PrincetonConfigV3, width) );
  confType.insert_native<uint32_t>( "height", offsetof(PrincetonConfigV3, height) );
  confType.insert_native<uint32_t>( "orgX", offsetof(PrincetonConfigV3, orgX) );
  confType.insert_native<uint32_t>( "orgY", offsetof(PrincetonConfigV3, orgY) );
  confType.insert_native<uint32_t>( "binX", offsetof(PrincetonConfigV3, binX) );
  confType.insert_native<uint32_t>( "binY", offsetof(PrincetonConfigV3, binY) );
  confType.insert_native<float>( "exposureTime", offsetof(PrincetonConfigV3, exposureTime) );
  confType.insert_native<float>( "coolingTemp", offsetof(PrincetonConfigV3, coolingTemp) );
  confType.insert_native<uint8_t>( "gainIndex", offsetof(PrincetonConfigV3, gainIndex) );
  confType.insert_native<uint8_t>( "readoutSpeedIndex", offsetof(PrincetonConfigV3, readoutSpeedIndex) );
  confType.insert_native<uint16_t>( "exposureEventCode", offsetof(PrincetonConfigV3, exposureEventCode) );
  confType.insert_native<uint32_t>( "numDelayShots", offsetof(PrincetonConfigV3, numDelayShots) );

  return confType ;
}

void
PrincetonConfigV3::store( const Pds::Princeton::ConfigV3& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  PrincetonConfigV3 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
