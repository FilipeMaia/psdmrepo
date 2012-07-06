#ifndef H5DATATYPES_EVRCONFIGDATA_H
#define H5DATATYPES_EVRCONFIGDATA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigData.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "hdf5pp/Type.h"
#include "pdsdata/evr/PulseConfig.hh"
#include "pdsdata/evr/PulseConfigV3.hh"
#include "pdsdata/evr/EventCodeV3.hh"
#include "pdsdata/evr/EventCodeV4.hh"
#include "pdsdata/evr/EventCodeV5.hh"
#include "pdsdata/evr/IOChannel.hh"
#include "pdsdata/evr/OutputMap.hh"
#include "pdsdata/evr/OutputMapV2.hh"
#include "pdsdata/evr/SequencerConfigV1.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace H5DataTypes {

//
// Helper type for Pds::EvrData::PulseConfig
//
class EvrPulseConfig {
public:

  EvrPulseConfig () {}
  EvrPulseConfig ( const Pds::EvrData::PulseConfig& pconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  uint32_t pulse;
  int16_t  trigger;
  int16_t  set;
  int16_t  clear;
  uint8_t  polarity;
  uint8_t  map_set_enable;
  uint8_t  map_reset_enable;
  uint8_t  map_trigger_enable;
  uint32_t prescale;
  uint32_t delay;
  uint32_t width;

};

//
// Helper type for Pds::EvrData::PulseConfigV3
//
class EvrPulseConfigV3 {
public:

  EvrPulseConfigV3 () {}
  EvrPulseConfigV3 ( const Pds::EvrData::PulseConfigV3& pconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  uint16_t  pulseId;
  uint16_t  polarity;
  uint32_t  prescale;
  uint32_t  delay;
  uint32_t  width;

};

//
// Helper type for Pds::EvrData::OutputMap
//
class EvrOutputMap {
public:

  EvrOutputMap () {}
  EvrOutputMap ( const Pds::EvrData::OutputMap& mconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type conn_type() ;

private:

  int16_t source ;
  int16_t source_id ;
  int16_t conn ;
  int16_t conn_id ;

};

//
// Helper type for Pds::EvrData::OutputMapV2
//
class EvrOutputMapV2 {
public:

  EvrOutputMapV2 () {}
  EvrOutputMapV2 ( const Pds::EvrData::OutputMapV2& mconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type conn_type() ;

private:

  int16_t source ;
  int16_t source_id ;
  int16_t conn ;
  int16_t conn_id ;
  int16_t module ;

};

//
// Helper type for Pds::EvrData::EventCodeV3
//
class EvrEventCodeV3 {
public:

  EvrEventCodeV3 () {}
  EvrEventCodeV3 ( const Pds::EvrData::EventCodeV3& evtcode ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  uint16_t  code;
  uint8_t   isReadout;
  uint8_t   isTerminator;
  uint32_t  maskTrigger;
  uint32_t  maskSet;
  uint32_t  maskClear;

};

//
// Helper type for Pds::EvrData::EventCodeV4
//
class EvrEventCodeV4 {
public:

  EvrEventCodeV4 () {}
  EvrEventCodeV4 ( const Pds::EvrData::EventCodeV4& evtcode ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  uint16_t  code;
  uint8_t   isReadout;
  uint8_t   isTerminator;
  uint32_t  reportDelay;
  uint32_t  reportWidth;
  uint32_t  maskTrigger;
  uint32_t  maskSet;
  uint32_t  maskClear;

};

//
// Helper type for Pds::EvrData::EventCodeV5
//
class EvrEventCodeV5 : boost::noncopyable {
public:

  EvrEventCodeV5 ();
  EvrEventCodeV5 ( const Pds::EvrData::EventCodeV5& evtcode ) ;
  ~EvrEventCodeV5 ();

  EvrEventCodeV5& operator= ( const Pds::EvrData::EventCodeV5& evtcode ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  uint16_t  code;
  uint8_t   isReadout;
  uint8_t   isCommand;
  uint8_t   isLatch;
  uint32_t  reportDelay;
  uint32_t  reportWidth;
  uint32_t  releaseCode;
  uint32_t  maskTrigger;
  uint32_t  maskSet;
  uint32_t  maskClear;
  char*     desc;

};

//
// Helper type for Pds::EvrData::IOChannel
//
struct EvrIOChannelDetInfo_Data {
  uint32_t processId;
  const char* detector;
  const char* device;
  uint32_t detId;
  uint32_t devId;
};

class EvrIOChannel : boost::noncopyable {
public:

  EvrIOChannel ();
  EvrIOChannel( const Pds::EvrData::IOChannel& chan );
  ~EvrIOChannel ();

  EvrIOChannel& operator=( const Pds::EvrData::IOChannel& chan );

  static hdf5pp::Type stored_type();
  static hdf5pp::Type native_type();

private:

  char*    name;
  size_t   ninfo;
  EvrIOChannelDetInfo_Data* info;

};

//
// Helper type for Pds::EvrData::SequencerConfigV1
//
struct EvrSequencerEntry_Data {
  uint32_t eventcode;
  uint32_t delay;
};

class EvrSequencerConfigV1 : boost::noncopyable {
public:

  EvrSequencerConfigV1 ();
  EvrSequencerConfigV1( const Pds::EvrData::SequencerConfigV1& chan );
  ~EvrSequencerConfigV1 ();

  static hdf5pp::Type stored_type();
  static hdf5pp::Type native_type();

private:

  uint16_t sync_source;
  uint16_t beam_source;
  uint32_t cycles;
  uint32_t length;
  size_t nentries;
  EvrSequencerEntry_Data* entries;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRCONFIGDATA_H
