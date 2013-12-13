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
  : fullScale(v.fullScale())
  , offset(v.offset())
  , coupling(v.coupling())
  , bandwidth(v.bandwidth())
{
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
  vertType.insert_native<double>( "fullScale", offsetof(AcqirisVertV1, fullScale) ) ;
  vertType.insert_native<double>( "offset", offsetof(AcqirisVertV1, offset) ) ;
  vertType.insert_native<uint32_t>( "coupling", offsetof(AcqirisVertV1, coupling) ) ;
  vertType.insert_native<uint32_t>( "bandwidth", offsetof(AcqirisVertV1, bandwidth) ) ;

  return vertType ;
}

AcqirisHorizV1::AcqirisHorizV1 ( const Pds::Acqiris::HorizV1& h )
  : sampInterval(h.sampInterval())
  , delayTime(h.delayTime())
  , nbrSamples(h.nbrSamples())
  , nbrSegments(h.nbrSegments())
{
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
  horizType.insert_native<double>( "sampInterval", offsetof(AcqirisHorizV1, sampInterval) ) ;
  horizType.insert_native<double>( "delayTime", offsetof(AcqirisHorizV1, delayTime) ) ;
  horizType.insert_native<uint32_t>( "nbrSamples", offsetof(AcqirisHorizV1, nbrSamples) ) ;
  horizType.insert_native<uint32_t>( "nbrSegments", offsetof(AcqirisHorizV1, nbrSegments) ) ;

  return horizType ;
}

AcqirisTrigV1::AcqirisTrigV1 ( const Pds::Acqiris::TrigV1& t )
  : coupling(t.coupling())
  , input(t.input())
  , slope(t.slope())
  , level(t.level())
{
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
  trigType.insert_native<uint32_t>( "coupling", offsetof(AcqirisTrigV1, coupling) ) ;
  trigType.insert_native<uint32_t>( "input", offsetof(AcqirisTrigV1, input) ) ;
  trigType.insert_native<uint32_t>( "slope", offsetof(AcqirisTrigV1, slope) ) ;
  trigType.insert_native<double>( "level", offsetof(AcqirisTrigV1, level) ) ;

  return trigType ;
}

AcqirisConfigV1::AcqirisConfigV1 ( const Pds::Acqiris::ConfigV1& c )
  : nbrConvertersPerChannel(c.nbrConvertersPerChannel())
  , channelMask(c.channelMask())
  , nbrChannels(c.nbrChannels())
  , nbrBanks(c.nbrBanks())
{
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
  confType.insert_native<uint32_t>( "nbrConvertersPerChannel", offsetof(AcqirisConfigV1, nbrConvertersPerChannel) ) ;
  confType.insert_native<uint32_t>( "channelMask", offsetof(AcqirisConfigV1, channelMask) ) ;
  confType.insert_native<uint32_t>( "nbrChannels", offsetof(AcqirisConfigV1, nbrChannels) ) ;
  confType.insert_native<uint32_t>( "nbrBanks", offsetof(AcqirisConfigV1, nbrBanks) ) ;

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
  const ndarray<const Pds::Acqiris::VertV1, 1>& pdsdata = config.vert();
  for ( uint32_t i = 0 ; i < nbrChannels ; ++ i ) {
    vdata[i] = AcqirisVertV1(pdsdata[i]);
  }
  storeDataObjects ( nbrChannels, vdata, "vert", grp ) ;

}

} // namespace H5DataTypes
