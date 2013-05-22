//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IpimbDataV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/IpimbDataV1.h"

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

IpimbDataV1::IpimbDataV1 ( const XtcType& data )
  : triggerCounter(data.triggerCounter())
  , config0(data.config0())
  , config1(data.config1())
  , config2(data.config2())
  , channel0(data.channel0())
  , channel1(data.channel1())
  , channel2(data.channel2())
  , channel3(data.channel3())
  , checksum(data.checksum())
  , channel0Volts(data.channel0Volts())
  , channel1Volts(data.channel1Volts())
  , channel2Volts(data.channel2Volts())
  , channel3Volts(data.channel3Volts())
{
}

hdf5pp::Type
IpimbDataV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
IpimbDataV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<IpimbDataV1>() ;
  type.insert_native<uint64_t>( "triggerCounter", offsetof(IpimbDataV1,triggerCounter) ) ;
  type.insert_native<uint16_t>( "config0", offsetof(IpimbDataV1,config0) ) ;
  type.insert_native<uint16_t>( "config1", offsetof(IpimbDataV1,config1) ) ;
  type.insert_native<uint16_t>( "config2", offsetof(IpimbDataV1,config2) ) ;
  type.insert_native<uint16_t>( "channel0", offsetof(IpimbDataV1,channel0) ) ;
  type.insert_native<uint16_t>( "channel1", offsetof(IpimbDataV1,channel1) ) ;
  type.insert_native<uint16_t>( "channel2", offsetof(IpimbDataV1,channel2) ) ;
  type.insert_native<uint16_t>( "channel3", offsetof(IpimbDataV1,channel3) ) ;
  type.insert_native<uint16_t>( "checksum", offsetof(IpimbDataV1,checksum) ) ;
  type.insert_native<float>( "channel0Volts", offsetof(IpimbDataV1,channel0Volts) ) ;
  type.insert_native<float>( "channel1Volts", offsetof(IpimbDataV1,channel1Volts) ) ;
  type.insert_native<float>( "channel2Volts", offsetof(IpimbDataV1,channel2Volts) ) ;
  type.insert_native<float>( "channel3Volts", offsetof(IpimbDataV1,channel3Volts) ) ;

  return type ;
}

} // namespace H5DataTypes
