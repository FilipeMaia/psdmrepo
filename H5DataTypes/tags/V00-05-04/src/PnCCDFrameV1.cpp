//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnCCDFrameV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PnCCDFrameV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/ArrayType.h"
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
PnCCDFrameV1::PnCCDFrameV1 ( const XtcType& frame )
  : specialWord(frame.specialWord())
  , frameNumber(frame.frameNumber())
  , timeStampHi(frame.timeStampHi())
  , timeStampLo(frame.timeStampLo())
{
}

hdf5pp::Type
PnCCDFrameV1::stored_type( const ConfigXtcType1& config )
{
  return native_type( config.numLinks() ) ;
}

hdf5pp::Type
PnCCDFrameV1::native_type( const ConfigXtcType1& config )
{
  return native_type( config.numLinks() ) ;
}

hdf5pp::Type
PnCCDFrameV1::stored_type( const ConfigXtcType2& config )
{
  return native_type( config.numLinks() ) ;
}

hdf5pp::Type
PnCCDFrameV1::native_type( const ConfigXtcType2& config )
{
  return native_type( config.numLinks() ) ;
}

hdf5pp::Type
PnCCDFrameV1::native_type( unsigned numlinks )
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<PnCCDFrameV1>() ;
  type.insert_native<uint32_t>( "specialWord", offsetof(PnCCDFrameV1,specialWord) ) ;
  type.insert_native<uint32_t>( "frameNumber", offsetof(PnCCDFrameV1,frameNumber) ) ;
  type.insert_native<uint32_t>( "timeStampHi", offsetof(PnCCDFrameV1,timeStampHi) ) ;
  type.insert_native<uint32_t>( "timeStampLo", offsetof(PnCCDFrameV1,timeStampLo) ) ;

  return hdf5pp::ArrayType::arrayType ( type, numlinks );
}

hdf5pp::Type
PnCCDFrameV1::stored_data_type( const ConfigXtcType1& config )
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<uint16_t>::native_type() ;

  // get few constants
  const uint32_t numLinks = config.numLinks() ;
  const unsigned sizeofData = (config.payloadSizePerLink()-sizeof(XtcType))/sizeof(uint16_t);

  hsize_t dims[] = { numLinks, sizeofData } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

hdf5pp::Type
PnCCDFrameV1::stored_data_type( const ConfigXtcType2& config )
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<uint16_t>::native_type() ;

  // get few constants
  const uint32_t numLinks = config.numLinks() ;
  const unsigned sizeofData = (config.payloadSizePerLink()-sizeof(XtcType))/sizeof(uint16_t);

  hsize_t dims[] = { numLinks, sizeofData } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

} // namespace H5DataTypes
