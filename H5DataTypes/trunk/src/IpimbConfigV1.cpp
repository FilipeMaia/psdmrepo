//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IpimbConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/IpimbConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
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

IpimbConfigV1::IpimbConfigV1 ( const Pds::Ipimb::ConfigV1& data )
  : triggerCounter(data.triggerCounter())
  , serialID(data.serialID())
  , chargeAmpRange(data.chargeAmpRange())
  , calibrationRange(data.calibrationRange())
  , resetLength(data.resetLength())
  , resetDelay(data.resetDelay())
  , chargeAmpRefVoltage(data.chargeAmpRefVoltage())
  , calibrationVoltage(data.calibrationVoltage())
  , diodeBias(data.diodeBias())
  , status(data.status())
  , errors(data.errors())
  , calStrobeLength(data.calStrobeLength())
  , trigDelay(data.trigDelay())
{
  for (unsigned ch = 0; ch != 4; ++ ch) {
    capacitorValue[ch] = data.capacitorValue(ch);
  }
}

hdf5pp::Type
IpimbConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
IpimbConfigV1::native_type()
{
  hdf5pp::EnumType<uint8_t> capValType = hdf5pp::EnumType<uint8_t>::enumType();
  capValType.insert("c_1pF", Pds::Ipimb::ConfigV1::c_1pF);
  capValType.insert("c_100pF", Pds::Ipimb::ConfigV1::c_100pF);
  capValType.insert("c_10nF", Pds::Ipimb::ConfigV1::c_10nF);

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<IpimbConfigV1>() ;
  confType.insert_native<uint64_t>( "triggerCounter", offsetof(IpimbConfigV1,triggerCounter) );
  confType.insert_native<uint64_t>( "serialID", offsetof(IpimbConfigV1,serialID) );
  confType.insert_native<uint16_t>( "chargeAmpRange", offsetof(IpimbConfigV1,chargeAmpRange) );
  confType.insert( "capacitorValue", offsetof(IpimbConfigV1,capacitorValue), capValType, 4 );
  confType.insert_native<uint16_t>( "calibrationRange", offsetof(IpimbConfigV1,calibrationRange) );
  confType.insert_native<uint32_t>( "resetLength", offsetof(IpimbConfigV1,resetLength) );
  confType.insert_native<uint16_t>( "resetDelay", offsetof(IpimbConfigV1,resetDelay) );
  confType.insert_native<float>( "chargeAmpRefVoltage", offsetof(IpimbConfigV1,chargeAmpRefVoltage) );
  confType.insert_native<float>( "calibrationVoltage", offsetof(IpimbConfigV1,calibrationVoltage) );
  confType.insert_native<float>( "diodeBias", offsetof(IpimbConfigV1,diodeBias) );
  confType.insert_native<uint16_t>( "status", offsetof(IpimbConfigV1,status) );
  confType.insert_native<uint16_t>( "errors", offsetof(IpimbConfigV1,errors) );
  confType.insert_native<uint16_t>( "calStrobeLength", offsetof(IpimbConfigV1,calStrobeLength) );
  confType.insert_native<uint32_t>( "trigDelay", offsetof(IpimbConfigV1,trigDelay) );

  return confType ;
}

void
IpimbConfigV1::store( const Pds::Ipimb::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  IpimbConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
