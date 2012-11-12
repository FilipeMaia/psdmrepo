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
{
  m_data.triggerCounter = data.triggerCounter();
  m_data.config0 = data.config0();
  m_data.config1 = data.config1();
  m_data.config2 = data.config2();
  m_data.channel0 = data.channel0();
  m_data.channel1 = data.channel1();
  m_data.channel2 = data.channel2();
  m_data.channel3 = data.channel3();
  m_data.channel0ps = data.channel0ps();
  m_data.channel1ps = data.channel1ps();
  m_data.channel2ps = data.channel2ps();
  m_data.channel3ps = data.channel3ps();
  m_data.checksum = data.checksum();
  m_data.channel0Volts = data.channel0Volts();
  m_data.channel1Volts = data.channel1Volts();
  m_data.channel2Volts = data.channel2Volts();
  m_data.channel3Volts = data.channel3Volts();
  m_data.channel0psVolts = data.channel0psVolts();
  m_data.channel1psVolts = data.channel1psVolts();
  m_data.channel2psVolts = data.channel2psVolts();
  m_data.channel3psVolts = data.channel3psVolts();
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
  type.insert_native<uint64_t>( "triggerCounter", offsetof(IpimbDataV2_Data,triggerCounter) ) ;
  type.insert_native<uint16_t>( "config0", offsetof(IpimbDataV2_Data,config0) ) ;
  type.insert_native<uint16_t>( "config1", offsetof(IpimbDataV2_Data,config1) ) ;
  type.insert_native<uint16_t>( "config2", offsetof(IpimbDataV2_Data,config2) ) ;
  type.insert_native<uint16_t>( "channel0", offsetof(IpimbDataV2_Data,channel0) ) ;
  type.insert_native<uint16_t>( "channel1", offsetof(IpimbDataV2_Data,channel1) ) ;
  type.insert_native<uint16_t>( "channel2", offsetof(IpimbDataV2_Data,channel2) ) ;
  type.insert_native<uint16_t>( "channel3", offsetof(IpimbDataV2_Data,channel3) ) ;
  type.insert_native<uint16_t>( "channel0ps", offsetof(IpimbDataV2_Data,channel0ps) ) ;
  type.insert_native<uint16_t>( "channel1ps", offsetof(IpimbDataV2_Data,channel1ps) ) ;
  type.insert_native<uint16_t>( "channel2ps", offsetof(IpimbDataV2_Data,channel2ps) ) ;
  type.insert_native<uint16_t>( "channel3ps", offsetof(IpimbDataV2_Data,channel3ps) ) ;
  type.insert_native<uint16_t>( "checksum", offsetof(IpimbDataV2_Data,checksum) ) ;
  type.insert_native<float>( "channel0Volts", offsetof(IpimbDataV2_Data,channel0Volts) ) ;
  type.insert_native<float>( "channel1Volts", offsetof(IpimbDataV2_Data,channel1Volts) ) ;
  type.insert_native<float>( "channel2Volts", offsetof(IpimbDataV2_Data,channel2Volts) ) ;
  type.insert_native<float>( "channel3Volts", offsetof(IpimbDataV2_Data,channel3Volts) ) ;
  type.insert_native<float>( "channel0psVolts", offsetof(IpimbDataV2_Data,channel0psVolts) ) ;
  type.insert_native<float>( "channel1psVolts", offsetof(IpimbDataV2_Data,channel1psVolts) ) ;
  type.insert_native<float>( "channel2psVolts", offsetof(IpimbDataV2_Data,channel2psVolts) ) ;
  type.insert_native<float>( "channel3psVolts", offsetof(IpimbDataV2_Data,channel3psVolts) ) ;

  return type ;
}

} // namespace H5DataTypes
