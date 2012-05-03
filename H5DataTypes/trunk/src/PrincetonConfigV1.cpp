//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PrincetonConfigV1.h"

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

PrincetonConfigV1::PrincetonConfigV1 ( const Pds::Princeton::ConfigV1& data )
  : width(data.width())
  , height(data.height())
  , orgX(data.orgX())
  , orgY(data.orgY())
  , binX(data.binX())
  , binY(data.binY())
  , exposureTime(data.exposureTime())
  , coolingTemp(data.coolingTemp())
  , readoutSpeedIndex(data.readoutSpeedIndex())
  , readoutEventCode(data.readoutEventCode())
  , delayMode(data.delayMode())
{
}

hdf5pp::Type
PrincetonConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PrincetonConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<PrincetonConfigV1>() ;
  confType.insert_native<uint32_t>( "width", offsetof(PrincetonConfigV1, width) );
  confType.insert_native<uint32_t>( "height", offsetof(PrincetonConfigV1, height) );
  confType.insert_native<uint32_t>( "orgX", offsetof(PrincetonConfigV1, orgX) );
  confType.insert_native<uint32_t>( "orgY", offsetof(PrincetonConfigV1, orgY) );
  confType.insert_native<uint32_t>( "binX", offsetof(PrincetonConfigV1, binX) );
  confType.insert_native<uint32_t>( "binY", offsetof(PrincetonConfigV1, binY) );
  confType.insert_native<float>( "exposureTime", offsetof(PrincetonConfigV1, exposureTime) );
  confType.insert_native<float>( "coolingTemp", offsetof(PrincetonConfigV1, coolingTemp) );
  confType.insert_native<uint32_t>( "readoutSpeedIndex", offsetof(PrincetonConfigV1, readoutSpeedIndex) );
  confType.insert_native<uint16_t>( "readoutEventCode", offsetof(PrincetonConfigV1, readoutEventCode) );
  confType.insert_native<uint16_t>( "delayMode", offsetof(PrincetonConfigV1, delayMode) );

  return confType ;
}

void
PrincetonConfigV1::store( const Pds::Princeton::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  PrincetonConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
