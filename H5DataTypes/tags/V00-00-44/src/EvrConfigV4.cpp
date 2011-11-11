//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrConfigV4...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EvrConfigV4.h"

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

EvrConfigV4::EvrConfigV4 ( const Pds::EvrData::ConfigV4& data )
{
  m_data.neventcodes = data.neventcodes() ;
  m_data.npulses = data.npulses() ;
  m_data.noutputs = data.noutputs() ;
}

hdf5pp::Type
EvrConfigV4::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrConfigV4::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<EvrConfigV4>() ;
  confType.insert_native<uint32_t>( "neventcodes", offsetof(EvrConfigV4_Data,neventcodes) ) ;
  confType.insert_native<uint32_t>( "npulses", offsetof(EvrConfigV4_Data,npulses) ) ;
  confType.insert_native<uint32_t>( "noutputs", offsetof(EvrConfigV4_Data,noutputs) ) ;

  return confType ;
}

void
EvrConfigV4::store( const Pds::EvrData::ConfigV4& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  EvrConfigV4 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // event codes
  const uint32_t neventcodes = config.neventcodes() ;
  EvrEventCodeV4 ecodes[neventcodes] ;
  for ( uint32_t i = 0 ; i < neventcodes ; ++ i ) {
    ecodes[i] = EvrEventCodeV4( config.eventcode(i) ) ;
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
  EvrOutputMap mdata[noutputs] ;
  for ( uint32_t i = 0 ; i < noutputs ; ++ i ) {
    mdata[i] = EvrOutputMap( config.output_map(i) ) ;
  }
  storeDataObjects ( noutputs, mdata, "output_maps", grp ) ;

}

} // namespace H5DataTypes
