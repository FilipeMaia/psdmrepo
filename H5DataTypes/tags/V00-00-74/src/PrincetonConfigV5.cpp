//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonConfigV5...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PrincetonConfigV5.h"

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

PrincetonConfigV5::PrincetonConfigV5 ( const Pds::Princeton::ConfigV5& data )
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
  , maskedHeight(data.maskedHeight())
  , kineticHeight(data.kineticHeight())
  , vsSpeed(data.vsSpeed())
  , infoReportInterval(data.infoReportInterval())
  , exposureEventCode(data.exposureEventCode())
  , numDelayShots(data.numDelayShots())
{
}

hdf5pp::Type
PrincetonConfigV5::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PrincetonConfigV5::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<PrincetonConfigV5>() ;
  confType.insert_native<uint32_t>( "width", offsetof(PrincetonConfigV5, width) );
  confType.insert_native<uint32_t>( "height", offsetof(PrincetonConfigV5, height) );
  confType.insert_native<uint32_t>( "orgX", offsetof(PrincetonConfigV5, orgX) );
  confType.insert_native<uint32_t>( "orgY", offsetof(PrincetonConfigV5, orgY) );
  confType.insert_native<uint32_t>( "binX", offsetof(PrincetonConfigV5, binX) );
  confType.insert_native<uint32_t>( "binY", offsetof(PrincetonConfigV5, binY) );
  confType.insert_native<float>( "exposureTime", offsetof(PrincetonConfigV5, exposureTime) );
  confType.insert_native<float>( "coolingTemp", offsetof(PrincetonConfigV5, coolingTemp) );
  confType.insert_native<uint16_t>( "gainIndex", offsetof(PrincetonConfigV5, gainIndex) );
  confType.insert_native<uint16_t>( "readoutSpeedIndex", offsetof(PrincetonConfigV5, readoutSpeedIndex) );
  confType.insert_native<uint32_t>( "maskedHeight", offsetof(PrincetonConfigV5, maskedHeight) );
  confType.insert_native<uint32_t>( "kineticHeight", offsetof(PrincetonConfigV5, kineticHeight) );
  confType.insert_native<float>( "vsSpeed", offsetof(PrincetonConfigV5, vsSpeed) );
  confType.insert_native<int16_t>( "infoReportInterval", offsetof(PrincetonConfigV5, infoReportInterval) );
  confType.insert_native<uint16_t>( "exposureEventCode", offsetof(PrincetonConfigV5, exposureEventCode) );
  confType.insert_native<uint32_t>( "numDelayShots", offsetof(PrincetonConfigV5, numDelayShots) );

  return confType ;
}

void
PrincetonConfigV5::store( const Pds::Princeton::ConfigV5& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  PrincetonConfigV5 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
