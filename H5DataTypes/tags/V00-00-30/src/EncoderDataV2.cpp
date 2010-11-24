//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EncoderDataV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EncoderDataV2.h"

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

//----------------
// Constructors --
//----------------
EncoderDataV2::EncoderDataV2 ( const XtcType& data )
{
  m_data._33mhz_timestamp = data._33mhz_timestamp ;
  m_data.encoder_count[0] = data._encoder_count[0] ;
  m_data.encoder_count[1] = data._encoder_count[1] ;
  m_data.encoder_count[2] = data._encoder_count[2] ;
}

hdf5pp::Type
EncoderDataV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EncoderDataV2::native_type()
{
  hdf5pp::ArrayType countType = hdf5pp::ArrayType::arrayType<uint32_t>(3);  

  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<EncoderDataV2>() ;
  type.insert_native<uint32_t>( "_33mhz_timestamp", offsetof(EncoderDataV2_Data,_33mhz_timestamp) ) ;
  type.insert( "encoder_count", offsetof(EncoderDataV2_Data,encoder_count), countType ) ;

  return type;
}

} // namespace H5DataTypes
