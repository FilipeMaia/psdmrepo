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

EvrConfigV1::EvrConfigV1 ( const Pds::EvrData::ConfigV1& data )
{
  m_data.npulses = data.npulses() ;
  m_data.noutputs = data.noutputs() ;
}

hdf5pp::Type
EvrConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<EvrConfigV1>() ;
  confType.insert_native<uint32_t>( "npulses", offsetof(EvrConfigV1_Data,npulses) ) ;
  confType.insert_native<uint32_t>( "noutputs", offsetof(EvrConfigV1_Data,noutputs) ) ;

  return confType ;
}

void
EvrConfigV1::store( const Pds::EvrData::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  EvrConfigV1 data ( config ) ;
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
