//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/AcqirisConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/H5DataUtils.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "hdf5pp/DataSet.h"
#include "hdf5pp/DataSpace.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

//----------------
// Constructors --
//----------------
AcqirisVertV1::AcqirisVertV1 ( const Pds::Acqiris::VertV1& v )
{
  m_data.fullScale = v.fullScale() ;
  m_data.offset = v.offset() ;
  m_data.coupling = v.coupling() ;
  m_data.bandwidth = v.bandwidth() ;
}

hdf5pp::Type
AcqirisVertV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AcqirisVertV1::native_type()
{
  hdf5pp::CompoundType vertType = hdf5pp::CompoundType::compoundType<AcqirisVertV1>() ;
  vertType.insert_native<double>( "fullScale", offsetof(AcqirisVertV1_Data,fullScale) ) ;
  vertType.insert_native<double>( "offset", offsetof(AcqirisVertV1_Data,offset) ) ;
  vertType.insert_native<uint32_t>( "coupling", offsetof(AcqirisVertV1_Data,coupling) ) ;
  vertType.insert_native<uint32_t>( "bandwidth", offsetof(AcqirisVertV1_Data,bandwidth) ) ;

  return vertType ;
}

AcqirisHorizV1::AcqirisHorizV1 ( const Pds::Acqiris::HorizV1& h )
{
  m_data.sampInterval = h.sampInterval() ;
  m_data.delayTime = h.delayTime() ;
  m_data.nbrSamples = h.nbrSamples() ;
  m_data.nbrSegments = h.nbrSegments() ;
}

hdf5pp::Type
AcqirisHorizV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AcqirisHorizV1::native_type()
{
  hdf5pp::CompoundType horizType = hdf5pp::CompoundType::compoundType<AcqirisHorizV1>() ;
  horizType.insert_native<double>( "sampInterval", offsetof(AcqirisHorizV1_Data,sampInterval) ) ;
  horizType.insert_native<double>( "delayTime", offsetof(AcqirisHorizV1_Data,delayTime) ) ;
  horizType.insert_native<uint32_t>( "nbrSamples", offsetof(AcqirisHorizV1_Data,nbrSamples) ) ;
  horizType.insert_native<uint32_t>( "nbrSegments", offsetof(AcqirisHorizV1_Data,nbrSegments) ) ;

  return horizType ;
}

AcqirisTrigV1::AcqirisTrigV1 ( const Pds::Acqiris::TrigV1& t )
{
  m_data.coupling = t.coupling() ;
  m_data.input = t.input() ;
  m_data.slope = t.slope() ;
  m_data.level = t.level() ;
}

hdf5pp::Type
AcqirisTrigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AcqirisTrigV1::native_type()
{
  hdf5pp::CompoundType trigType = hdf5pp::CompoundType::compoundType<AcqirisTrigV1>() ;
  trigType.insert_native<uint32_t>( "coupling", offsetof(AcqirisTrigV1_Data,coupling) ) ;
  trigType.insert_native<uint32_t>( "input", offsetof(AcqirisTrigV1_Data,input) ) ;
  trigType.insert_native<uint32_t>( "slope", offsetof(AcqirisTrigV1_Data,slope) ) ;
  trigType.insert_native<double>( "level", offsetof(AcqirisTrigV1_Data,level) ) ;

  return trigType ;
}

AcqirisConfigV1::AcqirisConfigV1 ( const Pds::Acqiris::ConfigV1& c )
{
  m_data.nbrConvertersPerChannel = c.nbrConvertersPerChannel() ;
  m_data.channelMask = c.channelMask() ;
  m_data.nbrChannels = c.nbrChannels() ;
  m_data.nbrBanks = c.nbrBanks() ;
}

hdf5pp::Type
AcqirisConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
AcqirisConfigV1::native_type()
{
  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<AcqirisConfigV1>() ;
  confType.insert_native<uint32_t>( "nbrConvertersPerChannel", offsetof(AcqirisConfigV1_Data,nbrConvertersPerChannel) ) ;
  confType.insert_native<uint32_t>( "channelMask", offsetof(AcqirisConfigV1_Data,channelMask) ) ;
  confType.insert_native<uint32_t>( "nbrChannels", offsetof(AcqirisConfigV1_Data,nbrChannels) ) ;
  confType.insert_native<uint32_t>( "nbrBanks", offsetof(AcqirisConfigV1_Data,nbrBanks) ) ;

  return confType ;
}

void
AcqirisConfigV1::store ( const Pds::Acqiris::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  AcqirisConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // make scalar data set for subobject
  AcqirisHorizV1 hdata ( config.horiz() ) ;
  storeDataObject ( hdata, "horiz", grp ) ;

  // make scalar data set for subobject
  AcqirisTrigV1 tdata ( config.trig() ) ;
  storeDataObject ( tdata, "trig", grp ) ;

  // make array data set for subobject
  const uint32_t nbrChannels = config.nbrChannels() ;
  AcqirisVertV1 vdata[nbrChannels] ;
  for ( uint32_t i = 0 ; i < nbrChannels ; ++ i ) {
    vdata[i] = AcqirisVertV1( config.vert(i) ) ;
  }
  storeDataObjects ( nbrChannels, vdata, "vert", grp ) ;

}

} // namespace H5DataTypes
