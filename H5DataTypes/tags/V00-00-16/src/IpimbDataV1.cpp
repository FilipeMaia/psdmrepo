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
#include "SITConfig/SITConfig.h"

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

IpimbDataV1::IpimbDataV1 ( const Pds::Ipimb::DataV1& data )
{
  m_data.triggerCounter = data.triggerCounter();
  m_data.config0 = data.config0();
  m_data.config1 = data.config1();
  m_data.config2 = data.config2();
  m_data.channel0 = data.channel0();
  m_data.channel1 = data.channel1();
  m_data.channel2 = data.channel2();
  m_data.channel3 = data.channel3();
  m_data.checksum = data.checksum();
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
  type.insert_native<uint64_t>( "triggerCounter", offsetof(IpimbDataV1_Data,triggerCounter) ) ;
  type.insert_native<uint16_t>( "config0", offsetof(IpimbDataV1_Data,config0) ) ;
  type.insert_native<uint16_t>( "config1", offsetof(IpimbDataV1_Data,config1) ) ;
  type.insert_native<uint16_t>( "config2", offsetof(IpimbDataV1_Data,config2) ) ;
  type.insert_native<uint16_t>( "channel0", offsetof(IpimbDataV1_Data,channel0) ) ;
  type.insert_native<uint16_t>( "channel1", offsetof(IpimbDataV1_Data,channel1) ) ;
  type.insert_native<uint16_t>( "channel2", offsetof(IpimbDataV1_Data,channel2) ) ;
  type.insert_native<uint16_t>( "channel3", offsetof(IpimbDataV1_Data,channel3) ) ;
  type.insert_native<uint16_t>( "checksum", offsetof(IpimbDataV1_Data,checksum) ) ;

  return type ;
}

} // namespace H5DataTypes
