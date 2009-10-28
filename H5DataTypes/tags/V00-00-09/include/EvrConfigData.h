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
#include "pdsdata/evr/OutputMap.hh"

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
struct EvrPulseConfig_Data {
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
} ;

class EvrPulseConfig {
public:

  EvrPulseConfig () {}
  EvrPulseConfig ( const Pds::EvrData::PulseConfig& pconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  EvrPulseConfig_Data m_data ;

};

//
// Helper type for Pds::EvrData::OutputMap
//
struct EvrOutputMap_Data {
  int16_t source ;
  int16_t source_id ;
  int16_t conn ;
  int16_t conn_id ;
};

class EvrOutputMap {
public:

  EvrOutputMap () {}
  EvrOutputMap ( const Pds::EvrData::OutputMap& mconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  EvrOutputMap_Data m_data ;

};


} // namespace H5DataTypes

#endif // H5DATATYPES_EVRCONFIGDATA_H
