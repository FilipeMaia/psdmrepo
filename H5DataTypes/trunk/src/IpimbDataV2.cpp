//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class IpimbDataV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/IpimbDataV2.h"

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

IpimbDataV2::IpimbDataV2 ( const XtcType& data )
  : triggerCounter(data.triggerCounter())
  , config0(data.config0())
  , config1(data.config1())
  , config2(data.config2())
  , channel0(data.channel0())
  , channel1(data.channel1())
  , channel2(data.channel2())
  , channel3(data.channel3())
  , channel0ps(data.channel0ps())
  , channel1ps(data.channel1ps())
  , channel2ps(data.channel2ps())
  , channel3ps(data.channel3ps())
  , checksum(data.checksum())
  , channel0Volts(data.channel0Volts())
  , channel1Volts(data.channel1Volts())
  , channel2Volts(data.channel2Volts())
  , channel3Volts(data.channel3Volts())
  , channel0psVolts(data.channel0psVolts())
  , channel1psVolts(data.channel1psVolts())
  , channel2psVolts(data.channel2psVolts())
  , channel3psVolts(data.channel3psVolts())
{
}

hdf5pp::Type
IpimbDataV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
IpimbDataV2::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<IpimbDataV2>() ;
  type.insert_native<uint64_t>( "triggerCounter", offsetof(IpimbDataV2,triggerCounter) ) ;
  type.insert_native<uint16_t>( "config0", offsetof(IpimbDataV2,config0) ) ;
  type.insert_native<uint16_t>( "config1", offsetof(IpimbDataV2,config1) ) ;
  type.insert_native<uint16_t>( "config2", offsetof(IpimbDataV2,config2) ) ;
  type.insert_native<uint16_t>( "channel0", offsetof(IpimbDataV2,channel0) ) ;
  type.insert_native<uint16_t>( "channel1", offsetof(IpimbDataV2,channel1) ) ;
  type.insert_native<uint16_t>( "channel2", offsetof(IpimbDataV2,channel2) ) ;
  type.insert_native<uint16_t>( "channel3", offsetof(IpimbDataV2,channel3) ) ;
  type.insert_native<uint16_t>( "channel0ps", offsetof(IpimbDataV2,channel0ps) ) ;
  type.insert_native<uint16_t>( "channel1ps", offsetof(IpimbDataV2,channel1ps) ) ;
  type.insert_native<uint16_t>( "channel2ps", offsetof(IpimbDataV2,channel2ps) ) ;
  type.insert_native<uint16_t>( "channel3ps", offsetof(IpimbDataV2,channel3ps) ) ;
  type.insert_native<uint16_t>( "checksum", offsetof(IpimbDataV2,checksum) ) ;
  type.insert_native<float>( "channel0Volts", offsetof(IpimbDataV2,channel0Volts) ) ;
  type.insert_native<float>( "channel1Volts", offsetof(IpimbDataV2,channel1Volts) ) ;
  type.insert_native<float>( "channel2Volts", offsetof(IpimbDataV2,channel2Volts) ) ;
  type.insert_native<float>( "channel3Volts", offsetof(IpimbDataV2,channel3Volts) ) ;
  type.insert_native<float>( "channel0psVolts", offsetof(IpimbDataV2,channel0psVolts) ) ;
  type.insert_native<float>( "channel1psVolts", offsetof(IpimbDataV2,channel1psVolts) ) ;
  type.insert_native<float>( "channel2psVolts", offsetof(IpimbDataV2,channel2psVolts) ) ;
  type.insert_native<float>( "channel3psVolts", offsetof(IpimbDataV2,channel3psVolts) ) ;

  return type ;
}

} // namespace H5DataTypes
