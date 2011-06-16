//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataIpimbV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataIpimbV1.h"

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

BldDataIpimbV1::BldDataIpimbV1 ( const XtcType& data )
{
  m_data.ipimbData.triggerCounter = data.ipimbData.triggerCounter();
  m_data.ipimbData.config0 = data.ipimbData.config0();
  m_data.ipimbData.config1 = data.ipimbData.config1();
  m_data.ipimbData.config2 = data.ipimbData.config2();
  m_data.ipimbData.channel0 = data.ipimbData.channel0();
  m_data.ipimbData.channel1 = data.ipimbData.channel1();
  m_data.ipimbData.channel2 = data.ipimbData.channel2();
  m_data.ipimbData.channel3 = data.ipimbData.channel3();
  m_data.ipimbData.channel0ps = data.ipimbData.channel0ps();
  m_data.ipimbData.channel1ps = data.ipimbData.channel1ps();
  m_data.ipimbData.channel2ps = data.ipimbData.channel2ps();
  m_data.ipimbData.channel3ps = data.ipimbData.channel3ps();
  m_data.ipimbData.checksum = data.ipimbData.checksum();
  m_data.ipimbData.channel0Volts = data.ipimbData.channel0Volts();
  m_data.ipimbData.channel1Volts = data.ipimbData.channel1Volts();
  m_data.ipimbData.channel2Volts = data.ipimbData.channel2Volts();
  m_data.ipimbData.channel3Volts = data.ipimbData.channel3Volts();
  m_data.ipimbData.channel0psVolts = data.ipimbData.channel0psVolts();
  m_data.ipimbData.channel1psVolts = data.ipimbData.channel1psVolts();
  m_data.ipimbData.channel2psVolts = data.ipimbData.channel2psVolts();
  m_data.ipimbData.channel3psVolts = data.ipimbData.channel3psVolts();

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
  m_data.ipimbConfig.trigPsDelay = data.ipimbConfig.trigPsDelay();
  m_data.ipimbConfig.adcDelay = data.ipimbConfig.adcDelay();

  std::copy( data.ipmFexData.channel, data.ipmFexData.channel+LusiIpmFexV1_Data::CHSIZE, 
      m_data.ipmFexData.channel);
  m_data.ipmFexData.sum = data.ipmFexData.sum;
  m_data.ipmFexData.xpos = data.ipmFexData.xpos;
  m_data.ipmFexData.ypos = data.ipmFexData.ypos;
}

BldDataIpimbV1::~BldDataIpimbV1 ()
{
}

hdf5pp::Type
BldDataIpimbV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataIpimbV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataIpimbV1_Data>() ;

  type.insert( "ipimbData", offsetof(BldDataIpimbV1_Data,ipimbData), IpimbDataV2::native_type() );
  type.insert( "ipimbConfig", offsetof(BldDataIpimbV1_Data,ipimbConfig), IpimbConfigV2::native_type() );
  type.insert( "ipmFexData", offsetof(BldDataIpimbV1_Data,ipmFexData), LusiIpmFexV1::native_type() );

  return type ;
}

} // namespace H5DataTypes
