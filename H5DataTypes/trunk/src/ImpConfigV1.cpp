//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImpConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/ImpConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
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

ImpConfigV1::ImpConfigV1 ( const XtcType& data )
  : range(data.range())
  , calRange(data.calRange())
  , reset(data.reset())
  , biasData(data.biasData())
  , calData(data.calData())
  , biasDacData(data.biasDacData())
  , calStrobe(data.calStrobe())
  , numberOfSamples(data.numberOfSamples())
  , trigDelay(data.trigDelay())
  , adcDelay(data.adcDelay())
{
}

hdf5pp::Type
ImpConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
ImpConfigV1::native_type()
{
  typedef ImpConfigV1 DsType;
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<DsType>();
  type.insert("range", offsetof(DsType, range), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("calRange", offsetof(DsType, calRange), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("reset", offsetof(DsType, reset), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("biasData", offsetof(DsType, biasData), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("calData", offsetof(DsType, calData), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("biasDacData", offsetof(DsType, biasDacData), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("calStrobe", offsetof(DsType, calStrobe), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("numberOfSamples", offsetof(DsType, numberOfSamples), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("trigDelay", offsetof(DsType, trigDelay), hdf5pp::TypeTraits<uint32_t>::stored_type());
  type.insert("adcDelay", offsetof(DsType, adcDelay), hdf5pp::TypeTraits<uint32_t>::stored_type());
  return type;
}

void
ImpConfigV1::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  ImpConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
