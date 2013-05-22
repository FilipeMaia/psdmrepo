//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV6...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EvrConfigV6.h"

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

EvrConfigV6::EvrConfigV6 ( const Pds::EvrData::ConfigV6& data )
  : neventcodes(data.neventcodes())
  , npulses(data.npulses())
  , noutputs(data.noutputs())
{
}

hdf5pp::Type
EvrConfigV6::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrConfigV6::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<EvrConfigV6>() ;
  confType.insert_native<uint32_t>( "neventcodes", offsetof(EvrConfigV6,neventcodes) ) ;
  confType.insert_native<uint32_t>( "npulses", offsetof(EvrConfigV6,npulses) ) ;
  confType.insert_native<uint32_t>( "noutputs", offsetof(EvrConfigV6,noutputs) ) ;

  return confType ;
}

void
EvrConfigV6::store( const Pds::EvrData::ConfigV6& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  EvrConfigV6 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // event codes
  const uint32_t neventcodes = config.neventcodes() ;
  EvrEventCodeV5 ecodes[neventcodes] ;
  for ( uint32_t i = 0 ; i < neventcodes ; ++ i ) {
    ecodes[i] = config.eventcode(i);
  }
  storeDataObjects ( neventcodes, ecodes, "eventcodes", grp ) ;

  // pulses data
  const uint32_t npulses = config.npulses() ;
  EvrPulseConfigV3 pdata[npulses] ;
  for ( uint32_t i = 0 ; i < npulses ; ++ i ) {
    pdata[i] = EvrPulseConfigV3( config.pulse(i) ) ;
  }
  storeDataObjects ( npulses, pdata, "pulses", grp ) ;

  // output map data
  const uint32_t noutputs = config.noutputs() ;
  EvrOutputMapV2 mdata[noutputs] ;
  for ( uint32_t i = 0 ; i < noutputs ; ++ i ) {
    mdata[i] = EvrOutputMapV2( config.output_map(i) ) ;
  }
  storeDataObjects ( noutputs, mdata, "output_maps", grp ) ;

  // sequences config
  EvrSequencerConfigV1 seq_config(config.seq_config());
  storeDataObject ( seq_config, "seq_config", grp ) ;

}

} // namespace H5DataTypes
