//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigData...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EvrConfigData.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/TypeTraits.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

//
// Helper type for Pds::EvrData::PulseConfig
//
EvrPulseConfig::EvrPulseConfig ( const Pds::EvrData::PulseConfig& pconfig )
{
  m_data.pulse = pconfig.pulse() ;
  m_data.trigger = pconfig.trigger() ;
  m_data.set = pconfig.set() ;
  m_data.clear = pconfig.clear() ;
  m_data.polarity = pconfig.polarity() ;
  m_data.map_set_enable = pconfig.map_set_enable() ;
  m_data.map_reset_enable = pconfig.map_reset_enable() ;
  m_data.map_trigger_enable = pconfig.map_trigger_enable() ;
  m_data.prescale = pconfig.prescale() ;
  m_data.delay = pconfig.delay() ;
  m_data.width = pconfig.width() ;
}

hdf5pp::Type
EvrPulseConfig::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrPulseConfig::native_type()
{
  hdf5pp::CompoundType pulseType = hdf5pp::CompoundType::compoundType< EvrPulseConfig_Data >() ;
  pulseType.insert_native<uint32_t>( "pulse", offsetof(EvrPulseConfig_Data,pulse) ) ;
  pulseType.insert_native<int16_t>( "trigger", offsetof(EvrPulseConfig_Data,trigger) ) ;
  pulseType.insert_native<int16_t>( "set", offsetof(EvrPulseConfig_Data,set) ) ;
  pulseType.insert_native<int16_t>( "clear", offsetof(EvrPulseConfig_Data,clear) ) ;
  pulseType.insert_native<uint8_t>( "polarity", offsetof(EvrPulseConfig_Data,polarity) ) ;
  pulseType.insert_native<uint8_t>( "map_set_enable", offsetof(EvrPulseConfig_Data,map_set_enable) ) ;
  pulseType.insert_native<uint8_t>( "map_reset_enable", offsetof(EvrPulseConfig_Data,map_reset_enable) ) ;
  pulseType.insert_native<uint8_t>( "map_trigger_enable", offsetof(EvrPulseConfig_Data,map_trigger_enable) ) ;
  pulseType.insert_native<uint32_t>( "prescale", offsetof(EvrPulseConfig_Data,prescale) ) ;
  pulseType.insert_native<uint32_t>( "delay", offsetof(EvrPulseConfig_Data,delay) ) ;
  pulseType.insert_native<uint32_t>( "width", offsetof(EvrPulseConfig_Data,width) ) ;

  return pulseType ;
}

//
// Helper type for Pds::EvrData::PulseConfigV3
//
EvrPulseConfigV3::EvrPulseConfigV3 ( const Pds::EvrData::PulseConfigV3& pconfig )
{
  m_data.pulseId = pconfig.pulseId() ;
  m_data.polarity = pconfig.polarity() ;
  m_data.prescale = pconfig.prescale() ;
  m_data.delay = pconfig.delay() ;
  m_data.width = pconfig.width() ;
}

hdf5pp::Type
EvrPulseConfigV3::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrPulseConfigV3::native_type()
{
  hdf5pp::CompoundType pulseType = hdf5pp::CompoundType::compoundType< EvrPulseConfigV3_Data >() ;
  pulseType.insert_native<uint16_t>( "pulseId", offsetof(EvrPulseConfigV3_Data,pulseId) ) ;
  pulseType.insert_native<uint16_t>( "polarity", offsetof(EvrPulseConfigV3_Data,polarity) ) ;
  pulseType.insert_native<uint32_t>( "prescale", offsetof(EvrPulseConfigV3_Data,prescale) ) ;
  pulseType.insert_native<uint32_t>( "delay", offsetof(EvrPulseConfigV3_Data,delay) ) ;
  pulseType.insert_native<uint32_t>( "width", offsetof(EvrPulseConfigV3_Data,width) ) ;

  return pulseType ;
}

//
// Helper type for Pds::EvrData::OutputMap
//
EvrOutputMap::EvrOutputMap ( const Pds::EvrData::OutputMap& mconfig )
{
  m_data.source = mconfig.source() ;
  m_data.source_id = mconfig.source_id() ;
  m_data.conn = mconfig.conn() ;
  m_data.conn_id = mconfig.conn_id() ;
}

hdf5pp::Type
EvrOutputMap::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrOutputMap::native_type()
{
  hdf5pp::EnumType<int16_t> srcEnumType = hdf5pp::EnumType<int16_t>::enumType() ;
  srcEnumType.insert ( "Pulse", Pds::EvrData::OutputMap::Pulse ) ;
  srcEnumType.insert ( "DBus", Pds::EvrData::OutputMap::DBus ) ;
  srcEnumType.insert ( "Prescaler", Pds::EvrData::OutputMap::Prescaler ) ;
  srcEnumType.insert ( "Force_High", Pds::EvrData::OutputMap::Force_High ) ;
  srcEnumType.insert ( "Force_Low", Pds::EvrData::OutputMap::Force_Low ) ;

  hdf5pp::EnumType<int16_t> connEnumType = hdf5pp::EnumType<int16_t>::enumType() ;
  connEnumType.insert ( "FrontPanel", Pds::EvrData::OutputMap::FrontPanel ) ;
  connEnumType.insert ( "UnivIO", Pds::EvrData::OutputMap::UnivIO ) ;

  hdf5pp::CompoundType mapType = hdf5pp::CompoundType::compoundType< EvrOutputMap_Data >() ;
  mapType.insert( "source", offsetof(EvrOutputMap_Data,source), srcEnumType ) ;
  mapType.insert_native<int16_t>( "source_id", offsetof(EvrOutputMap_Data,source_id) ) ;
  mapType.insert( "conn", offsetof(EvrOutputMap_Data,conn), connEnumType ) ;
  mapType.insert_native<int16_t>( "conn_id", offsetof(EvrOutputMap_Data,conn_id) ) ;

  return mapType ;
}

//
// Helper type for Pds::EvrData::EventCodeV3
//
EvrEventCodeV3::EvrEventCodeV3 ( const Pds::EvrData::EventCodeV3& evtcode )
{
  m_data.code = evtcode.code();
  m_data.isReadout = evtcode.isReadout();
  m_data.isTerminator = evtcode.isTerminator();
  m_data.maskTrigger = evtcode.maskTrigger();
  m_data.maskSet = evtcode.maskSet();
  m_data.maskClear = evtcode.maskClear();
}

hdf5pp::Type
EvrEventCodeV3::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrEventCodeV3::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType< EvrEventCodeV3_Data >() ;
  type.insert_native<uint16_t>( "code", offsetof(EvrEventCodeV3_Data,code) ) ;
  type.insert_native<uint8_t>( "isReadout", offsetof(EvrEventCodeV3_Data,isReadout) ) ;
  type.insert_native<uint8_t>( "isTerminator", offsetof(EvrEventCodeV3_Data,isTerminator) ) ;
  type.insert_native<uint32_t>( "maskTrigger", offsetof(EvrEventCodeV3_Data,maskTrigger) ) ;
  type.insert_native<uint32_t>( "maskSet", offsetof(EvrEventCodeV3_Data,maskSet) ) ;
  type.insert_native<uint32_t>( "maskClear", offsetof(EvrEventCodeV3_Data,maskClear) ) ;

  return type ;
}

} // namespace H5DataTypes
