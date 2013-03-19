//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AndorConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/AndorConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

AndorConfigV1::AndorConfigV1 ( const Pds::Andor::ConfigV1& data )
  : width(data.width())
  , height(data.height())
  , orgX(data.orgX())
  , orgY(data.orgY())
  , binX(data.binX())
  , binY(data.binY())
  , exposureTime(data.exposureTime())
  , coolingTemp(data.coolingTemp())
  , fanMode(data.fanMode())
  , baselineClamp(data.baselineClamp())
  , highCapacity(data.highCapacity())
  , gainIndex(data.gainIndex())
  , readoutSpeedIndex(data.readoutSpeedIndex())
  , exposureEventCode(data.exposureEventCode())
  , numDelayShots(data.numDelayShots())
{
}

hdf5pp::Type
AndorConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AndorConfigV1::native_type()
{
  hdf5pp::EnumType<uint8_t> fanModeEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  fanModeEnum.insert ( "ENUM_FAN_FULL", Pds::Andor::ConfigV1::ENUM_FAN_FULL ) ;
  fanModeEnum.insert ( "ENUM_FAN_LOW", Pds::Andor::ConfigV1::ENUM_FAN_LOW ) ;
  fanModeEnum.insert ( "ENUM_FAN_OFF", Pds::Andor::ConfigV1::ENUM_FAN_OFF ) ;
  fanModeEnum.insert ( "ENUM_FAN_ACQOFF", Pds::Andor::ConfigV1::ENUM_FAN_ACQOFF ) ;
  fanModeEnum.insert ( "ENUM_FAN_NUM", Pds::Andor::ConfigV1::ENUM_FAN_NUM ) ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<AndorConfigV1>() ;
  confType.insert_native<uint32_t>( "width", offsetof(AndorConfigV1, width) );
  confType.insert_native<uint32_t>( "height", offsetof(AndorConfigV1, height) );
  confType.insert_native<uint32_t>( "orgX", offsetof(AndorConfigV1, orgX) );
  confType.insert_native<uint32_t>( "orgY", offsetof(AndorConfigV1, orgY) );
  confType.insert_native<uint32_t>( "binX", offsetof(AndorConfigV1, binX) );
  confType.insert_native<uint32_t>( "binY", offsetof(AndorConfigV1, binY) );
  confType.insert_native<float>( "exposureTime", offsetof(AndorConfigV1, exposureTime) );
  confType.insert_native<float>( "coolingTemp", offsetof(AndorConfigV1, coolingTemp) );
  confType.insert( "fanMode", offsetof(AndorConfigV1, fanMode), fanModeEnum );
  confType.insert_native<uint8_t>( "baselineClamp", offsetof(AndorConfigV1, baselineClamp) );
  confType.insert_native<uint8_t>( "highCapacity", offsetof(AndorConfigV1, highCapacity) );
  confType.insert_native<uint8_t>( "gainIndex", offsetof(AndorConfigV1, gainIndex) );
  confType.insert_native<uint8_t>( "readoutSpeedIndex", offsetof(AndorConfigV1, readoutSpeedIndex) );
  confType.insert_native<uint16_t>( "exposureEventCode", offsetof(AndorConfigV1, exposureEventCode) );
  confType.insert_native<uint32_t>( "numDelayShots", offsetof(AndorConfigV1, numDelayShots) );

  return confType ;
}

void
AndorConfigV1::store( const Pds::Andor::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  AndorConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
