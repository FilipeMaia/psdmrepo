//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PrincetonConfigV2.h"

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

PrincetonConfigV2::PrincetonConfigV2 ( const Pds::Princeton::ConfigV2& data )
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
  , readoutEventCode(data.readoutEventCode())
  , delayMode(data.delayMode())
{
}

hdf5pp::Type
PrincetonConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PrincetonConfigV2::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<PrincetonConfigV2>() ;
  confType.insert_native<uint32_t>( "width", offsetof(PrincetonConfigV2, width) );
  confType.insert_native<uint32_t>( "height", offsetof(PrincetonConfigV2, height) );
  confType.insert_native<uint32_t>( "orgX", offsetof(PrincetonConfigV2, orgX) );
  confType.insert_native<uint32_t>( "orgY", offsetof(PrincetonConfigV2, orgY) );
  confType.insert_native<uint32_t>( "binX", offsetof(PrincetonConfigV2, binX) );
  confType.insert_native<uint32_t>( "binY", offsetof(PrincetonConfigV2, binY) );
  confType.insert_native<float>( "exposureTime", offsetof(PrincetonConfigV2, exposureTime) );
  confType.insert_native<float>( "coolingTemp", offsetof(PrincetonConfigV2, coolingTemp) );
  confType.insert_native<uint16_t>( "gainIndex", offsetof(PrincetonConfigV2, gainIndex) );
  confType.insert_native<uint16_t>( "readoutSpeedIndex", offsetof(PrincetonConfigV2, readoutSpeedIndex) );
  confType.insert_native<uint16_t>( "readoutEventCode", offsetof(PrincetonConfigV2, readoutEventCode) );
  confType.insert_native<uint16_t>( "delayMode", offsetof(PrincetonConfigV2, delayMode) );

  return confType ;
}

void
PrincetonConfigV2::store( const Pds::Princeton::ConfigV2& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  PrincetonConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
