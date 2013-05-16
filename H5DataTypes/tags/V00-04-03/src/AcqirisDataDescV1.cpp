//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisDataDescV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/AcqirisDataDescV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {


AcqirisTimestampV1::AcqirisTimestampV1(const XtcType& xtcData)
  : value(xtcData.value())
  , pos(xtcData.pos())
{
}

hdf5pp::Type 
AcqirisTimestampV1::stored_type() 
{
  return native_type();
}

hdf5pp::Type 
AcqirisTimestampV1::native_type() 
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<AcqirisTimestampV1>() ;
  type.insert_native<uint64_t>( "value", offsetof(AcqirisTimestampV1, value) ) ;
  type.insert_native<double>( "pos", offsetof(AcqirisTimestampV1, pos) ) ;
  
  return type;
}


//----------------
// Constructors --
//----------------
AcqirisDataDescV1::AcqirisDataDescV1 (const AcqirisDataDescV1::XtcType& xtcData)
  : nbrSamplesInSeg(xtcData.nbrSamplesInSeg())
  , nbrSegments(xtcData.nbrSegments())
  , indexFirstPoint(const_cast<AcqirisDataDescV1::XtcType&>(xtcData).indexFirstPoint())
{
}

hdf5pp::Type 
AcqirisDataDescV1::stored_type(const Pds::Acqiris::ConfigV1& config) 
{
  return native_type(config);
}

hdf5pp::Type 
AcqirisDataDescV1::native_type(const Pds::Acqiris::ConfigV1& config) 
{
  hdf5pp::CompoundType baseType = hdf5pp::CompoundType::compoundType<AcqirisDataDescV1>() ;
  baseType.insert_native<uint32_t>( "nbrSamplesInSeg", offsetof(AcqirisDataDescV1, nbrSamplesInSeg) ) ;
  baseType.insert_native<uint32_t>( "nbrSegments", offsetof(AcqirisDataDescV1, nbrSegments) ) ;
  baseType.insert_native<uint32_t>( "indexFirstPoint", offsetof(AcqirisDataDescV1, indexFirstPoint) ) ;

  hdf5pp::Type type = hdf5pp::ArrayType::arrayType ( baseType, config.nbrChannels() );

  return type ;
}

hdf5pp::Type
AcqirisDataDescV1::timestampType( const Pds::Acqiris::ConfigV1& config )
{
  const Pds::Acqiris::HorizV1& hconfig = config.horiz() ;

  hdf5pp::Type baseType = AcqirisTimestampV1::native_type() ;

  hsize_t dims[] = { config.nbrChannels(), hconfig.nbrSegments() } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

hdf5pp::Type
AcqirisDataDescV1::waveformType( const Pds::Acqiris::ConfigV1& config )
{
  const Pds::Acqiris::HorizV1& hconfig = config.horiz() ;

  hdf5pp::Type baseType = hdf5pp::TypeTraits<int16_t>::native_type() ;

  hsize_t dims[] = { config.nbrChannels(), hconfig.nbrSegments(), hconfig.nbrSamples() } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 3, dims );
}

} // namespace H5DataTypes
