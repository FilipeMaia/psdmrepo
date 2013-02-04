//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IpimbConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/IpimbConfigV2.h"

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

IpimbConfigV2::IpimbConfigV2 ( const Pds::Ipimb::ConfigV2& data )
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
  , trigPsDelay(data.trigPsDelay())
  , adcDelay(data.adcDelay())
{
  for (unsigned ch = 0; ch != 4; ++ ch) {
    capacitorValue[ch] = data.capacitorValue(ch);
  }
}

hdf5pp::Type
IpimbConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
IpimbConfigV2::native_type()
{
  hdf5pp::EnumType<uint8_t> capValType = hdf5pp::EnumType<uint8_t>::enumType();
  capValType.insert("c_1pF", Pds::Ipimb::ConfigV2::c_1pF);
  capValType.insert("c_4p7pF", Pds::Ipimb::ConfigV2::c_4p7pF);
  capValType.insert("c_24pF", Pds::Ipimb::ConfigV2::c_24pF);
  capValType.insert("c_120pF", Pds::Ipimb::ConfigV2::c_120pF);
  capValType.insert("c_620pF", Pds::Ipimb::ConfigV2::c_620pF);
  capValType.insert("c_3p3nF", Pds::Ipimb::ConfigV2::c_3p3nF);
  capValType.insert("c_10nF", Pds::Ipimb::ConfigV2::c_10nF);
  capValType.insert("expert", Pds::Ipimb::ConfigV2::expert);
  hdf5pp::Type capValArrType = hdf5pp::ArrayType::arrayType(capValType, 4);

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<IpimbConfigV2>() ;
  confType.insert_native<uint64_t>( "triggerCounter", offsetof(IpimbConfigV2,triggerCounter) );
  confType.insert_native<uint64_t>( "serialID", offsetof(IpimbConfigV2,serialID) );
  confType.insert_native<uint16_t>( "chargeAmpRange", offsetof(IpimbConfigV2,chargeAmpRange) );
  confType.insert( "capacitorValue", offsetof(IpimbConfigV2,capacitorValue), capValArrType );
  confType.insert_native<uint16_t>( "calibrationRange", offsetof(IpimbConfigV2,calibrationRange) );
  confType.insert_native<uint32_t>( "resetLength", offsetof(IpimbConfigV2,resetLength) );
  confType.insert_native<uint16_t>( "resetDelay", offsetof(IpimbConfigV2,resetDelay) );
  confType.insert_native<float>( "chargeAmpRefVoltage", offsetof(IpimbConfigV2,chargeAmpRefVoltage) );
  confType.insert_native<float>( "calibrationVoltage", offsetof(IpimbConfigV2,calibrationVoltage) );
  confType.insert_native<float>( "diodeBias", offsetof(IpimbConfigV2,diodeBias) );
  confType.insert_native<uint16_t>( "status", offsetof(IpimbConfigV2,status) );
  confType.insert_native<uint16_t>( "errors", offsetof(IpimbConfigV2,errors) );
  confType.insert_native<uint16_t>( "calStrobeLength", offsetof(IpimbConfigV2,calStrobeLength) );
  confType.insert_native<uint32_t>( "trigDelay", offsetof(IpimbConfigV2,trigDelay) );
  confType.insert_native<uint32_t>( "trigPsDelay", offsetof(IpimbConfigV2,trigPsDelay) );
  confType.insert_native<uint32_t>( "adcDelay", offsetof(IpimbConfigV2,adcDelay) );

  return confType ;
}

void
IpimbConfigV2::store( const Pds::Ipimb::ConfigV2& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  IpimbConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
