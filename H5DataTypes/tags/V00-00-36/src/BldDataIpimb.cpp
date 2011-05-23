//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataIpimb...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataIpimb.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

BldDataIpimb::BldDataIpimb ( const XtcType& data )
{
  m_data.ipimbData.triggerCounter = data.ipimbData.triggerCounter();
  m_data.ipimbData.config0 = data.ipimbData.config0();
  m_data.ipimbData.config1 = data.ipimbData.config1();
  m_data.ipimbData.config2 = data.ipimbData.config2();
  m_data.ipimbData.channel0 = data.ipimbData.channel0();
  m_data.ipimbData.channel1 = data.ipimbData.channel1();
  m_data.ipimbData.channel2 = data.ipimbData.channel2();
  m_data.ipimbData.channel3 = data.ipimbData.channel3();
  m_data.ipimbData.checksum = data.ipimbData.checksum();
  m_data.ipimbData.channel0Volts = data.ipimbData.channel0Volts();
  m_data.ipimbData.channel1Volts = data.ipimbData.channel1Volts();
  m_data.ipimbData.channel2Volts = data.ipimbData.channel2Volts();
  m_data.ipimbData.channel3Volts = data.ipimbData.channel3Volts();

  m_data.ipimbConfig.triggerCounter = data.ipimbConfig.triggerCounter();
  m_data.ipimbConfig.serialID = data.ipimbConfig.serialID();
  m_data.ipimbConfig.chargeAmpRange = data.ipimbConfig.chargeAmpRange();
  m_data.ipimbConfig.calibrationRange = data.ipimbConfig.calibrationRange();
  m_data.ipimbConfig.resetLength = data.ipimbConfig.resetLength();
  m_data.ipimbConfig.resetDelay = data.ipimbConfig.resetDelay();
  m_data.ipimbConfig.chargeAmpRefVoltage = data.ipimbConfig.chargeAmpRefVoltage();
  m_data.ipimbConfig.calibrationVoltage = data.ipimbConfig.calibrationVoltage();
  m_data.ipimbConfig.diodeBias = data.ipimbConfig.diodeBias();
  m_data.ipimbConfig.status = data.ipimbConfig.status();
  m_data.ipimbConfig.errors = data.ipimbConfig.errors();
  m_data.ipimbConfig.calStrobeLength = data.ipimbConfig.calStrobeLength();
  m_data.ipimbConfig.trigDelay = data.ipimbConfig.trigDelay();

  std::copy( data.ipmFexData.channel, data.ipmFexData.channel+LusiIpmFexV1_Data::CHSIZE, 
      m_data.ipmFexData.channel);
  m_data.ipmFexData.sum = data.ipmFexData.sum;
  m_data.ipmFexData.xpos = data.ipmFexData.xpos;
  m_data.ipmFexData.ypos = data.ipmFexData.ypos;
}

BldDataIpimb::~BldDataIpimb ()
{
}

hdf5pp::Type
BldDataIpimb::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataIpimb::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataIpimb_Data>() ;

  type.insert( "ipimbData", offsetof(BldDataIpimb_Data,ipimbData), IpimbDataV1::native_type() );
  type.insert( "ipimbConfig", offsetof(BldDataIpimb_Data,ipimbConfig), IpimbConfigV1::native_type() );
  type.insert( "ipmFexData", offsetof(BldDataIpimb_Data,ipmFexData), LusiIpmFexV1::native_type() );

  return type ;
}

} // namespace H5DataTypes
