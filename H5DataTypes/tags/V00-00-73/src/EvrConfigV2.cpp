//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EvrConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/EvrConfigData.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

EvrConfigV2::EvrConfigV2 ( const Pds::EvrData::ConfigV2& data )
{
  m_data.beam = data.beam() ;
  m_data.rate = data.rate() ;
  m_data.npulses = data.npulses() ;
  m_data.noutputs = data.noutputs() ;
}

hdf5pp::Type
EvrConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrConfigV2::native_type()
{
  hdf5pp::EnumType<int16_t> rateEnumType = hdf5pp::EnumType<int16_t>::enumType() ;
  rateEnumType.insert ( "r120Hz", Pds::EvrData::ConfigV2::r120Hz ) ;
  rateEnumType.insert ( "r60Hz", Pds::EvrData::ConfigV2::r60Hz ) ;
  rateEnumType.insert ( "r30Hz", Pds::EvrData::ConfigV2::r30Hz ) ;
  rateEnumType.insert ( "r10Hz", Pds::EvrData::ConfigV2::r10Hz ) ;
  rateEnumType.insert ( "r5Hz", Pds::EvrData::ConfigV2::r5Hz ) ;
  rateEnumType.insert ( "r1Hz", Pds::EvrData::ConfigV2::r1Hz ) ;
  rateEnumType.insert ( "r0_5Hz", Pds::EvrData::ConfigV2::r0_5Hz ) ;
  rateEnumType.insert ( "Single", Pds::EvrData::ConfigV2::Single ) ;

  hdf5pp::EnumType<int16_t> beamEnumType = hdf5pp::EnumType<int16_t>::enumType() ;
  beamEnumType.insert ( "Off", Pds::EvrData::ConfigV2::Off ) ;
  beamEnumType.insert ( "On", Pds::EvrData::ConfigV2::On ) ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<EvrConfigV2>() ;
  confType.insert( "beam", offsetof(EvrConfigV2_Data,beam), beamEnumType ) ;
  confType.insert( "rate", offsetof(EvrConfigV2_Data,rate), rateEnumType ) ;
  confType.insert_native<uint32_t>( "npulses", offsetof(EvrConfigV2_Data,npulses) ) ;
  confType.insert_native<uint32_t>( "noutputs", offsetof(EvrConfigV2_Data,noutputs) ) ;

  return confType ;
}

void
EvrConfigV2::store( const Pds::EvrData::ConfigV2& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  EvrConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // pulses data
  const uint32_t npulses = config.npulses() ;
  EvrPulseConfig pdata[npulses] ;
  for ( uint32_t i = 0 ; i < npulses ; ++ i ) {
    pdata[i] = EvrPulseConfig( config.pulse(i) ) ;
  }
  storeDataObjects ( npulses, pdata, "pulses", grp ) ;

  // output map data
  const uint32_t noutputs = config.noutputs() ;
  EvrOutputMap mdata[noutputs] ;
  for ( uint32_t i = 0 ; i < noutputs ; ++ i ) {
    mdata[i] = EvrOutputMap( config.output_map(i) ) ;
  }
  storeDataObjects ( noutputs, mdata, "output_maps", grp ) ;

}

} // namespace H5DataTypes
