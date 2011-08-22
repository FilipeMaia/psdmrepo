//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrIOConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EvrIOConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

EvrIOConfigV1::EvrIOConfigV1 ( const Pds::EvrData::IOConfigV1& data )
{
  m_data.conn = data.conn() ;
}

hdf5pp::Type
EvrIOConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EvrIOConfigV1::native_type()
{
  // type for conn enum
  hdf5pp::Type connEnumType = EvrOutputMap::conn_type() ;

  // config type
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<EvrIOConfigV1_Data>() ;
  confType.insert( "conn", offsetof(EvrIOConfigV1_Data, conn), connEnumType ) ;

  return confType ;
}

void
EvrIOConfigV1::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  EvrIOConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // channels
  const uint32_t nchannels = config.nchannels() ;
  EvrIOChannel channels[nchannels] ;
  for ( uint32_t i = 0 ; i < nchannels ; ++ i ) {
    channels[i] = config.channel(i);
  }
  storeDataObjects ( nchannels, channels, "channels", grp ) ;
}

} // namespace H5DataTypes
