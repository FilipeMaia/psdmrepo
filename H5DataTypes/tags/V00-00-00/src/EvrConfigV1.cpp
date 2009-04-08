//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EvrConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

//----------------
// Constructors --
//----------------
EvrPulseConfigV1::EvrPulseConfigV1 ( const Pds::EvrData::PulseConfig pconfig )
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
EvrPulseConfigV1::persType()
{
  hdf5pp::CompoundType pulseType = hdf5pp::CompoundType::compoundType< EvrPulseConfigV1_Data >() ;
  pulseType.insert( "pulse", offsetof(EvrPulseConfigV1_Data,pulse), hdf5pp::AtomicType::atomicType<uint32_t>() ) ;
  pulseType.insert( "trigger", offsetof(EvrPulseConfigV1_Data,trigger), hdf5pp::AtomicType::atomicType<int16_t>() ) ;
  pulseType.insert( "set", offsetof(EvrPulseConfigV1_Data,set), hdf5pp::AtomicType::atomicType<int16_t>() ) ;
  pulseType.insert( "clear", offsetof(EvrPulseConfigV1_Data,clear), hdf5pp::AtomicType::atomicType<int16_t>() ) ;
  pulseType.insert( "polarity", offsetof(EvrPulseConfigV1_Data,polarity), hdf5pp::AtomicType::atomicType<uint8_t>() ) ;
  pulseType.insert( "map_set_enable", offsetof(EvrPulseConfigV1_Data,map_set_enable), hdf5pp::AtomicType::atomicType<uint8_t>() ) ;
  pulseType.insert( "map_reset_enable", offsetof(EvrPulseConfigV1_Data,map_reset_enable), hdf5pp::AtomicType::atomicType<uint8_t>() ) ;
  pulseType.insert( "map_trigger_enable", offsetof(EvrPulseConfigV1_Data,map_trigger_enable), hdf5pp::AtomicType::atomicType<uint8_t>() ) ;
  pulseType.insert( "prescale", offsetof(EvrPulseConfigV1_Data,prescale), hdf5pp::AtomicType::atomicType<uint32_t>() ) ;
  pulseType.insert( "delay", offsetof(EvrPulseConfigV1_Data,delay), hdf5pp::AtomicType::atomicType<uint32_t>() ) ;
  pulseType.insert( "width", offsetof(EvrPulseConfigV1_Data,width), hdf5pp::AtomicType::atomicType<uint32_t>() ) ;

  return pulseType ;
}

EvrOutputMapV1::EvrOutputMapV1 ( const Pds::EvrData::OutputMap mconfig )
{
  m_data.source = mconfig.source() ;
  m_data.source_id = mconfig.source_id() ;
  m_data.conn = mconfig.conn() ;
  m_data.conn_id = mconfig.conn_id() ;
}

hdf5pp::Type
EvrOutputMapV1::persType()
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

  hdf5pp::CompoundType mapType = hdf5pp::CompoundType::compoundType< EvrOutputMapV1_Data >() ;
  mapType.insert( "source", offsetof(EvrOutputMapV1_Data,source), srcEnumType ) ;
  mapType.insert( "source_id", offsetof(EvrOutputMapV1_Data,source_id), hdf5pp::AtomicType::atomicType<int16_t>() ) ;
  mapType.insert( "conn", offsetof(EvrOutputMapV1_Data,conn), connEnumType ) ;
  mapType.insert( "conn_id", offsetof(EvrOutputMapV1_Data,conn_id), hdf5pp::AtomicType::atomicType<int16_t>() ) ;

  return mapType ;
}

EvrConfigV1::EvrConfigV1 ( const Pds::EvrData::ConfigV1& data )
{
  m_data.npulses = data.npulses() ;
  m_data.noutputs = data.noutputs() ;
}

hdf5pp::Type
EvrConfigV1::persType()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<EvrConfigV1>() ;
  confType.insert( "npulses", offsetof(EvrConfigV1_Data,npulses), hdf5pp::AtomicType::atomicType<uint32_t>() ) ;
  confType.insert( "noutputs", offsetof(EvrConfigV1_Data,noutputs), hdf5pp::AtomicType::atomicType<uint32_t>() ) ;

  return confType ;
}

void
storeEvrConfigV1( const Pds::EvrData::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  EvrConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // pulses data
  const uint32_t npulses = config.npulses() ;
  EvrPulseConfigV1 pdata[npulses] ;
  for ( uint32_t i = 0 ; i < npulses ; ++ i ) {
    pdata[i] = EvrPulseConfigV1( config.pulse(i) ) ;
  }
  storeDataObjects ( npulses, pdata, "pulses", grp ) ;

  // output map data
  const uint32_t noutputs = config.noutputs() ;
  EvrOutputMapV1 mdata[noutputs] ;
  for ( uint32_t i = 0 ; i < noutputs ; ++ i ) {
    mdata[i] = EvrOutputMapV1( config.output_map(i) ) ;
  }
  storeDataObjects ( noutputs, mdata, "output_maps", grp ) ;

}

} // namespace H5DataTypes
