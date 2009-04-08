#ifndef H5DATATYPES_EVRCONFIGV1_H
#define H5DATATYPES_EVRCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV1.
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
#include "hdf5pp/Group.h"
#include "pdsdata/evr/ConfigV1.hh"

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
struct EvrPulseConfigV1_Data {
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

class EvrPulseConfigV1 {
public:

  EvrPulseConfigV1 () {}
  EvrPulseConfigV1 ( const Pds::EvrData::PulseConfig pconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  EvrPulseConfigV1_Data m_data ;

};

//
// Helper type for Pds::EvrData::OutputMap
//
struct EvrOutputMapV1_Data {
  int16_t source ;
  int16_t source_id ;
  int16_t conn ;
  int16_t conn_id ;
};

class EvrOutputMapV1 {
public:

  EvrOutputMapV1 () {}
  EvrOutputMapV1 ( const Pds::EvrData::OutputMap mconfig ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  EvrOutputMapV1_Data m_data ;

};

//
// Helper type for Pds::EvrData::ConfigV1
//
struct EvrConfigV1_Data {
  uint32_t npulses;
  uint32_t noutputs;
};

class EvrConfigV1  {
public:

  EvrConfigV1 () {}
  EvrConfigV1 ( const Pds::EvrData::ConfigV1& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:

  EvrConfigV1_Data m_data ;

};

// store single config object at specified location
void storeEvrConfigV1( const Pds::EvrData::ConfigV1& config, hdf5pp::Group location ) ;

} // namespace H5DataTypes

#endif // H5DATATYPES_EVRCONFIGV1_H
