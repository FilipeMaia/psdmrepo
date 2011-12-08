//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EncoderDataV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EncoderDataV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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
EncoderDataV1::EncoderDataV1 ( const XtcType& data )
  : _33mhz_timestamp(data._33mhz_timestamp)
  , encoder_count(data._encoder_count)
{
}

hdf5pp::Type
EncoderDataV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EncoderDataV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<EncoderDataV1>() ;
  type.insert_native<uint32_t>( "_33mhz_timestamp", offsetof(EncoderDataV1, _33mhz_timestamp) ) ;
  type.insert_native<uint32_t>( "encoder_count", offsetof(EncoderDataV1, encoder_count) ) ;

  return type;
}

} // namespace H5DataTypes
