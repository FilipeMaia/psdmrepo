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
#include "hdf5pp/VlenType.h"
#include "hdf5pp/TypeTraits.h"
#include "pdsdata/evr/SequencerEntry.hh"

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
  : pulse(pconfig.pulse())
  , trigger(pconfig.trigger())
  , set(pconfig.set())
  , clear(pconfig.clear())
  , polarity(pconfig.polarity())
  , map_set_enable(pconfig.map_set_enable())
  , map_reset_enable(pconfig.map_reset_enable())
  , map_trigger_enable(pconfig.map_trigger_enable())
  , prescale(pconfig.prescale())
  , delay(pconfig.delay())
  , width(pconfig.width())
{
}

hdf5pp::Type
EvrPulseConfig::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrPulseConfig::native_type()
{
  hdf5pp::CompoundType pulseType = hdf5pp::CompoundType::compoundType< EvrPulseConfig >() ;
  pulseType.insert_native<uint32_t>( "pulse", offsetof(EvrPulseConfig,pulse) ) ;
  pulseType.insert_native<int16_t>( "trigger", offsetof(EvrPulseConfig,trigger) ) ;
  pulseType.insert_native<int16_t>( "set", offsetof(EvrPulseConfig,set) ) ;
  pulseType.insert_native<int16_t>( "clear", offsetof(EvrPulseConfig,clear) ) ;
  pulseType.insert_native<uint8_t>( "polarity", offsetof(EvrPulseConfig,polarity) ) ;
  pulseType.insert_native<uint8_t>( "map_set_enable", offsetof(EvrPulseConfig,map_set_enable) ) ;
  pulseType.insert_native<uint8_t>( "map_reset_enable", offsetof(EvrPulseConfig,map_reset_enable) ) ;
  pulseType.insert_native<uint8_t>( "map_trigger_enable", offsetof(EvrPulseConfig,map_trigger_enable) ) ;
  pulseType.insert_native<uint32_t>( "prescale", offsetof(EvrPulseConfig,prescale) ) ;
  pulseType.insert_native<uint32_t>( "delay", offsetof(EvrPulseConfig,delay) ) ;
  pulseType.insert_native<uint32_t>( "width", offsetof(EvrPulseConfig,width) ) ;

  return pulseType ;
}

//
// Helper type for Pds::EvrData::PulseConfigV3
//
EvrPulseConfigV3::EvrPulseConfigV3 ( const Pds::EvrData::PulseConfigV3& pconfig )
  : pulseId(pconfig.pulseId())
  , polarity(pconfig.polarity())
  , prescale(pconfig.prescale())
  , delay(pconfig.delay())
  , width(pconfig.width())
{
}

hdf5pp::Type
EvrPulseConfigV3::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrPulseConfigV3::native_type()
{
  hdf5pp::CompoundType pulseType = hdf5pp::CompoundType::compoundType< EvrPulseConfigV3 >() ;
  pulseType.insert_native<uint16_t>( "pulseId", offsetof(EvrPulseConfigV3,pulseId) ) ;
  pulseType.insert_native<uint16_t>( "polarity", offsetof(EvrPulseConfigV3,polarity) ) ;
  pulseType.insert_native<uint32_t>( "prescale", offsetof(EvrPulseConfigV3,prescale) ) ;
  pulseType.insert_native<uint32_t>( "delay", offsetof(EvrPulseConfigV3,delay) ) ;
  pulseType.insert_native<uint32_t>( "width", offsetof(EvrPulseConfigV3,width) ) ;

  return pulseType ;
}

//
// Helper type for Pds::EvrData::OutputMap
//
EvrOutputMap::EvrOutputMap ( const Pds::EvrData::OutputMap& mconfig )
  : source(mconfig.source())
  , source_id(mconfig.source_id())
  , conn(mconfig.conn())
  , conn_id(mconfig.conn_id())
{
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

  hdf5pp::Type connEnumType = conn_type() ;

  hdf5pp::CompoundType mapType = hdf5pp::CompoundType::compoundType< EvrOutputMap >() ;
  mapType.insert( "source", offsetof(EvrOutputMap,source), srcEnumType ) ;
  mapType.insert_native<int16_t>( "source_id", offsetof(EvrOutputMap,source_id) ) ;
  mapType.insert( "conn", offsetof(EvrOutputMap,conn), connEnumType ) ;
  mapType.insert_native<int16_t>( "conn_id", offsetof(EvrOutputMap,conn_id) ) ;

  return mapType ;
}

hdf5pp::Type
EvrOutputMap::conn_type()
{
  hdf5pp::EnumType<int16_t> connEnumType = hdf5pp::EnumType<int16_t>::enumType() ;
  connEnumType.insert ( "FrontPanel", Pds::EvrData::OutputMap::FrontPanel ) ;
  connEnumType.insert ( "UnivIO", Pds::EvrData::OutputMap::UnivIO ) ;

  return connEnumType;
}


//
// Helper type for Pds::EvrData::OutputMap
//
EvrOutputMapV2::EvrOutputMapV2 ( const Pds::EvrData::OutputMapV2& mconfig )
  : source(mconfig.source())
  , source_id(mconfig.source_id())
  , conn(mconfig.conn())
  , conn_id(mconfig.conn_id())
  , module(mconfig.module())
{
}

hdf5pp::Type
EvrOutputMapV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrOutputMapV2::native_type()
{
  hdf5pp::EnumType<int16_t> srcEnumType = hdf5pp::EnumType<int16_t>::enumType() ;
  srcEnumType.insert ( "Pulse", Pds::EvrData::OutputMapV2::Pulse ) ;
  srcEnumType.insert ( "DBus", Pds::EvrData::OutputMapV2::DBus ) ;
  srcEnumType.insert ( "Prescaler", Pds::EvrData::OutputMapV2::Prescaler ) ;
  srcEnumType.insert ( "Force_High", Pds::EvrData::OutputMapV2::Force_High ) ;
  srcEnumType.insert ( "Force_Low", Pds::EvrData::OutputMapV2::Force_Low ) ;

  hdf5pp::Type connEnumType = conn_type() ;

  hdf5pp::CompoundType mapType = hdf5pp::CompoundType::compoundType< EvrOutputMapV2 >() ;
  mapType.insert( "source", offsetof(EvrOutputMapV2,source), srcEnumType ) ;
  mapType.insert_native<int16_t>( "source_id", offsetof(EvrOutputMapV2, source_id) ) ;
  mapType.insert( "conn", offsetof(EvrOutputMapV2,conn), connEnumType ) ;
  mapType.insert_native<int16_t>( "conn_id", offsetof(EvrOutputMapV2, conn_id) ) ;
  mapType.insert_native<int16_t>( "module", offsetof(EvrOutputMapV2, module) ) ;

  return mapType ;
}

hdf5pp::Type
EvrOutputMapV2::conn_type()
{
  hdf5pp::EnumType<int16_t> connEnumType = hdf5pp::EnumType<int16_t>::enumType() ;
  connEnumType.insert ( "FrontPanel", Pds::EvrData::OutputMapV2::FrontPanel ) ;
  connEnumType.insert ( "UnivIO", Pds::EvrData::OutputMapV2::UnivIO ) ;

  return connEnumType;
}


//
// Helper type for Pds::EvrData::EventCodeV3
//
EvrEventCodeV3::EvrEventCodeV3 ( const Pds::EvrData::EventCodeV3& evtcode )
  : code(evtcode.code())
  , isReadout(evtcode.isReadout())
  , isTerminator(evtcode.isTerminator())
  , maskTrigger(evtcode.maskTrigger())
  , maskSet(evtcode.maskSet())
  , maskClear(evtcode.maskClear())
{
}

hdf5pp::Type
EvrEventCodeV3::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrEventCodeV3::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType< EvrEventCodeV3 >() ;
  type.insert_native<uint16_t>( "code", offsetof(EvrEventCodeV3,code) ) ;
  type.insert_native<uint8_t>( "isReadout", offsetof(EvrEventCodeV3,isReadout) ) ;
  type.insert_native<uint8_t>( "isTerminator", offsetof(EvrEventCodeV3,isTerminator) ) ;
  type.insert_native<uint32_t>( "maskTrigger", offsetof(EvrEventCodeV3,maskTrigger) ) ;
  type.insert_native<uint32_t>( "maskSet", offsetof(EvrEventCodeV3,maskSet) ) ;
  type.insert_native<uint32_t>( "maskClear", offsetof(EvrEventCodeV3,maskClear) ) ;

  return type ;
}

//
// Helper type for Pds::EvrData::EventCodeV4
//
EvrEventCodeV4::EvrEventCodeV4 ( const Pds::EvrData::EventCodeV4& evtcode )
  : code(evtcode.code())
  , isReadout(evtcode.isReadout())
  , isTerminator(evtcode.isTerminator())
  , reportDelay(evtcode.reportDelay())
  , reportWidth(evtcode.reportWidth())
  , maskTrigger(evtcode.maskTrigger())
  , maskSet(evtcode.maskSet())
  , maskClear(evtcode.maskClear())
{
}

hdf5pp::Type
EvrEventCodeV4::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrEventCodeV4::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType< EvrEventCodeV4 >() ;
  type.insert_native<uint16_t>( "code", offsetof(EvrEventCodeV4,code) ) ;
  type.insert_native<uint8_t>( "isReadout", offsetof(EvrEventCodeV4,isReadout) ) ;
  type.insert_native<uint8_t>( "isTerminator", offsetof(EvrEventCodeV4,isTerminator) ) ;
  type.insert_native<uint32_t>( "reportDelay", offsetof(EvrEventCodeV4,reportDelay) ) ;
  type.insert_native<uint32_t>( "reportWidth", offsetof(EvrEventCodeV4,reportWidth) ) ;
  type.insert_native<uint32_t>( "maskTrigger", offsetof(EvrEventCodeV4,maskTrigger) ) ;
  type.insert_native<uint32_t>( "maskSet", offsetof(EvrEventCodeV4,maskSet) ) ;
  type.insert_native<uint32_t>( "maskClear", offsetof(EvrEventCodeV4,maskClear) ) ;

  return type ;
}

//
// Helper type for Pds::EvrData::EventCodeV5
//
EvrEventCodeV5::EvrEventCodeV5 ()
  : desc(0)
{
}

EvrEventCodeV5::EvrEventCodeV5 ( const Pds::EvrData::EventCodeV5& evtcode )
  : code(evtcode.code())
  , isReadout(evtcode.isReadout())
  , isCommand(evtcode.isCommand())
  , isLatch(evtcode.isLatch())
  , reportDelay(evtcode.reportDelay())
  , reportWidth(evtcode.reportWidth())
  , releaseCode(evtcode.releaseCode())
  , maskTrigger(evtcode.maskTrigger())
  , maskSet(evtcode.maskSet())
  , maskClear(evtcode.maskClear())
{  
  const char* p = evtcode.desc();
  int len = strlen(p)+1;
  desc = new char[len];
  std::copy(p, p+len, desc);
}

EvrEventCodeV5&
EvrEventCodeV5::operator= ( const Pds::EvrData::EventCodeV5& evtcode )
{
  code = evtcode.code();
  isReadout = evtcode.isReadout();
  isCommand = evtcode.isCommand();
  isLatch = evtcode.isLatch();
  reportDelay = evtcode.reportDelay();
  reportWidth = evtcode.reportWidth();
  releaseCode = evtcode.releaseCode();
  maskTrigger = evtcode.maskTrigger();
  maskSet = evtcode.maskSet();
  maskClear = evtcode.maskClear();
  
  delete [] desc;
  const char* p = evtcode.desc();
  int len = strlen(p)+1;
  desc = new char[len];
  std::copy(p, p+len, desc);
  
  return *this;
}

EvrEventCodeV5::~EvrEventCodeV5 ()
{
  delete [] desc;
}

hdf5pp::Type
EvrEventCodeV5::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrEventCodeV5::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType< EvrEventCodeV5 >() ;
  type.insert_native<uint16_t>( "code", offsetof(EvrEventCodeV5,code) ) ;
  type.insert_native<uint8_t>( "isReadout", offsetof(EvrEventCodeV5,isReadout) ) ;
  type.insert_native<uint8_t>( "isCommand", offsetof(EvrEventCodeV5,isCommand) ) ;
  type.insert_native<uint8_t>( "isLatch", offsetof(EvrEventCodeV5,isLatch) ) ;
  type.insert_native<uint32_t>( "reportDelay", offsetof(EvrEventCodeV5,reportDelay) ) ;
  type.insert_native<uint32_t>( "reportWidth", offsetof(EvrEventCodeV5,reportWidth) ) ;
  type.insert_native<uint32_t>( "releaseCode", offsetof(EvrEventCodeV5,releaseCode) ) ;
  type.insert_native<uint32_t>( "maskTrigger", offsetof(EvrEventCodeV5,maskTrigger) ) ;
  type.insert_native<uint32_t>( "maskSet", offsetof(EvrEventCodeV5,maskSet) ) ;
  type.insert_native<uint32_t>( "maskClear", offsetof(EvrEventCodeV5,maskClear) ) ;
  type.insert_native<const char*>( "desc", offsetof(EvrEventCodeV5,desc) ) ;

  return type ;
}

//
// Helper type for Pds::EvrData::IOChannel
//

EvrIOChannel::EvrIOChannel ()
  : name(0)
  , info(0)
{
}

EvrIOChannel::EvrIOChannel ( const Pds::EvrData::IOChannel& chan )
  : name(0)
  , ninfo(chan.ninfo())
  , info(0)
{

  const char* p = chan.name();
  int len = strlen(p)+1;
  name = new char[len];
  std::copy(p, p+len, name);
  
  info = new EvrIOChannelDetInfo_Data[ninfo];
  for ( size_t i = 0 ; i != ninfo ; ++ i ) {
    const Pds::DetInfo& detinfo = chan.info(i);
    info[i].processId = detinfo.processId();
    info[i].detector = Pds::DetInfo::name(detinfo.detector());
    info[i].device = Pds::DetInfo::name(detinfo.device());
    info[i].detId = detinfo.devId();
    info[i].devId = detinfo.devId();
  }
}

EvrIOChannel&
EvrIOChannel::operator= ( const Pds::EvrData::IOChannel& chan )
{
  delete [] name;
  delete [] info;

  name = 0;
  ninfo = chan.ninfo();
  info = 0;

  const char* p = chan.name();
  int len = strlen(p)+1;
  name = new char[len];
  std::copy(p, p+len, name);
  
  info = new EvrIOChannelDetInfo_Data[ninfo];
  for ( size_t i = 0 ; i != ninfo ; ++ i ) {
    const Pds::DetInfo& detinfo = chan.info(i);
    info[i].processId = detinfo.processId();
    info[i].detector = Pds::DetInfo::name(detinfo.detector());
    info[i].device = Pds::DetInfo::name(detinfo.device());
    info[i].detId = detinfo.devId();
    info[i].devId = detinfo.devId();
  }
  
  return *this;
}

EvrIOChannel::~EvrIOChannel ()
{
  delete [] name;
  delete [] info;
}

hdf5pp::Type
EvrIOChannel::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrIOChannel::native_type()
{
  // DetInfo type
  hdf5pp::CompoundType detInfoType = hdf5pp::CompoundType::compoundType< EvrIOChannelDetInfo_Data >() ;
  detInfoType.insert_native<uint32_t>( "processId", offsetof(EvrIOChannelDetInfo_Data, processId) ) ;
  detInfoType.insert_native<const char*>( "detector", offsetof(EvrIOChannelDetInfo_Data, detector) ) ;
  detInfoType.insert_native<const char*>( "device", offsetof(EvrIOChannelDetInfo_Data, device) ) ;
  detInfoType.insert_native<uint32_t>( "detId", offsetof(EvrIOChannelDetInfo_Data, detId) ) ;
  detInfoType.insert_native<uint32_t>( "devId", offsetof(EvrIOChannelDetInfo_Data, devId) ) ;

  // variable-size type for info array
  hdf5pp::Type infoType = hdf5pp::VlenType::vlenType ( detInfoType );

  // Channel type
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType< EvrIOChannel >() ;
  type.insert_native<const char*>( "name", offsetof(EvrIOChannel, name) ) ;
  type.insert( "info", offsetof(EvrIOChannel, ninfo), infoType ) ;

  return type ;
}

//
// Helper type for Pds::EvrData::SequencerConfigV1
//

EvrSequencerConfigV1::EvrSequencerConfigV1 ()
  : entries(0)
{
}

EvrSequencerConfigV1::EvrSequencerConfigV1 ( const Pds::EvrData::SequencerConfigV1& data )
  : sync_source(data.sync_source())
  , beam_source(data.beam_source())
  , cycles(data.cycles())
  , length(data.length())
  , nentries(data.length())
{

  entries = new EvrSequencerEntry_Data[nentries];
  for ( size_t i = 0 ; i != nentries ; ++ i ) {
    const Pds::EvrData::SequencerEntry& entry = data.entry(i);
    entries[i].eventcode = entry.eventcode();
    entries[i].delay = entry.delay();
  }
}

EvrSequencerConfigV1::~EvrSequencerConfigV1 ()
{
  delete [] entries;
}

hdf5pp::Type
EvrSequencerConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrSequencerConfigV1::native_type()
{
  // SequencerEntry type
  hdf5pp::CompoundType entryType = hdf5pp::CompoundType::compoundType< EvrSequencerEntry_Data >() ;
  entryType.insert_native<uint32_t>( "eventcode", offsetof(EvrSequencerEntry_Data, eventcode) ) ;
  entryType.insert_native<uint32_t>( "delay", offsetof(EvrSequencerEntry_Data, delay) ) ;

  // variable-size type for info array
  hdf5pp::Type entriesType = hdf5pp::VlenType::vlenType ( entryType );

  hdf5pp::EnumType<int16_t> srcEnumType = hdf5pp::EnumType<int16_t>::enumType() ;
  srcEnumType.insert ( "r120Hz", Pds::EvrData::SequencerConfigV1::r120Hz ) ;
  srcEnumType.insert ( "r60Hz", Pds::EvrData::SequencerConfigV1::r60Hz ) ;
  srcEnumType.insert ( "r30Hz", Pds::EvrData::SequencerConfigV1::r30Hz ) ;
  srcEnumType.insert ( "r10Hz", Pds::EvrData::SequencerConfigV1::r10Hz ) ;
  srcEnumType.insert ( "r5Hz", Pds::EvrData::SequencerConfigV1::r5Hz ) ;
  srcEnumType.insert ( "r1Hz", Pds::EvrData::SequencerConfigV1::r1Hz ) ;
  srcEnumType.insert ( "r0_5Hz", Pds::EvrData::SequencerConfigV1::r0_5Hz ) ;
  srcEnumType.insert ( "Disable", Pds::EvrData::SequencerConfigV1::Disable ) ;

  // SequencerConfigV1 type
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType< EvrSequencerConfigV1 >() ;
  type.insert( "sync_source", offsetof(EvrSequencerConfigV1, sync_source), srcEnumType ) ;
  type.insert( "beam_source", offsetof(EvrSequencerConfigV1, beam_source), srcEnumType ) ;
  type.insert_native<uint32_t>( "cycles", offsetof(EvrSequencerConfigV1, cycles) ) ;
  type.insert_native<uint32_t>( "length", offsetof(EvrSequencerConfigV1, length) ) ;
  type.insert( "entries", offsetof(EvrSequencerConfigV1, nentries), entriesType ) ;

  return type ;
}

} // namespace H5DataTypes
