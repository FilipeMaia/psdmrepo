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

IpimbConfigV1::IpimbConfigV1 ( const Pds::Ipimb::ConfigV1& data )
{
  m_data.triggerCounter = data.triggerCounter();
  m_data.serialID = data.serialID();
  m_data.chargeAmpRange = data.chargeAmpRange();
  m_data.calibrationRange = data.calibrationRange();
  m_data.resetLength = data.resetLength();
  m_data.resetDelay = data.resetDelay();
  m_data.chargeAmpRefVoltage = data.chargeAmpRefVoltage();
  m_data.calibrationVoltage = data.calibrationVoltage();
  m_data.diodeBias = data.diodeBias();
  m_data.status = data.status();
  m_data.errors = data.errors();
  m_data.calStrobeLength = data.calStrobeLength();
  m_data.trigDelay = data.trigDelay();
}

hdf5pp::Type
IpimbConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
IpimbConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<IpimbConfigV1>() ;
  confType.insert_native<uint64_t>( "triggerCounter", offsetof(IpimbConfigV1_Data,triggerCounter) );
  confType.insert_native<uint64_t>( "serialID", offsetof(IpimbConfigV1_Data,serialID) );
  confType.insert_native<uint16_t>( "chargeAmpRange", offsetof(IpimbConfigV1_Data,chargeAmpRange) );
  confType.insert_native<uint16_t>( "calibrationRange", offsetof(IpimbConfigV1_Data,calibrationRange) );
  confType.insert_native<uint32_t>( "resetLength", offsetof(IpimbConfigV1_Data,resetLength) );
  confType.insert_native<uint16_t>( "resetDelay", offsetof(IpimbConfigV1_Data,resetDelay) );
  confType.insert_native<float>( "chargeAmpRefVoltage", offsetof(IpimbConfigV1_Data,chargeAmpRefVoltage) );
  confType.insert_native<float>( "calibrationVoltage", offsetof(IpimbConfigV1_Data,calibrationVoltage) );
  confType.insert_native<float>( "diodeBias", offsetof(IpimbConfigV1_Data,diodeBias) );
  confType.insert_native<uint16_t>( "status", offsetof(IpimbConfigV1_Data,status) );
  confType.insert_native<uint16_t>( "errors", offsetof(IpimbConfigV1_Data,errors) );
  confType.insert_native<uint16_t>( "calStrobeLength", offsetof(IpimbConfigV1_Data,calStrobeLength) );
  confType.insert_native<uint32_t>( "trigDelay", offsetof(IpimbConfigV1_Data,trigDelay) );

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
