//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonConfigV4...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PrincetonConfigV4.h"

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

PrincetonConfigV4::PrincetonConfigV4 ( const Pds::Princeton::ConfigV4& data )
  : width(data.width())
  , height(data.height())
  , orgX(data.orgX())
  , orgY(data.orgY())
  , binX(data.binX())
  , binY(data.binY())
  , maskedHeight(data.maskedHeight())
  , kineticHeight(data.kineticHeight())
  , vsSpeed(data.vsSpeed())
  , exposureTime(data.exposureTime())
  , coolingTemp(data.coolingTemp())
  , gainIndex(data.gainIndex())
  , readoutSpeedIndex(data.readoutSpeedIndex())
  , exposureEventCode(data.exposureEventCode())
  , numDelayShots(data.numDelayShots())
{
}

hdf5pp::Type
PrincetonConfigV4::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PrincetonConfigV4::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<PrincetonConfigV4>() ;
  confType.insert_native<uint32_t>( "width", offsetof(PrincetonConfigV4, width) );
  confType.insert_native<uint32_t>( "height", offsetof(PrincetonConfigV4, height) );
  confType.insert_native<uint32_t>( "orgX", offsetof(PrincetonConfigV4, orgX) );
  confType.insert_native<uint32_t>( "orgY", offsetof(PrincetonConfigV4, orgY) );
  confType.insert_native<uint32_t>( "binX", offsetof(PrincetonConfigV4, binX) );
  confType.insert_native<uint32_t>( "binY", offsetof(PrincetonConfigV4, binY) );
  confType.insert_native<uint32_t>( "maskedHeight", offsetof(PrincetonConfigV4, maskedHeight) );
  confType.insert_native<uint32_t>( "kineticHeight", offsetof(PrincetonConfigV4, kineticHeight) );
  confType.insert_native<float>( "vsSpeed", offsetof(PrincetonConfigV4, vsSpeed) );
  confType.insert_native<float>( "exposureTime", offsetof(PrincetonConfigV4, exposureTime) );
  confType.insert_native<float>( "coolingTemp", offsetof(PrincetonConfigV4, coolingTemp) );
  confType.insert_native<uint8_t>( "gainIndex", offsetof(PrincetonConfigV4, gainIndex) );
  confType.insert_native<uint8_t>( "readoutSpeedIndex", offsetof(PrincetonConfigV4, readoutSpeedIndex) );
  confType.insert_native<uint16_t>( "exposureEventCode", offsetof(PrincetonConfigV4, exposureEventCode) );
  confType.insert_native<uint32_t>( "numDelayShots", offsetof(PrincetonConfigV4, numDelayShots) );

  return confType ;
}

void
PrincetonConfigV4::store( const Pds::Princeton::ConfigV4& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  PrincetonConfigV4 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
