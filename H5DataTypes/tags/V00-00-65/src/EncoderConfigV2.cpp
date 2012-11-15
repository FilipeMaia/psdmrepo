//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EncoderConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EncoderConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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

EncoderConfigV2::EncoderConfigV2 ( const Pds::Encoder::ConfigV2& data )
  : chan_mask(data._chan_mask)
  , count_mode(data._count_mode)
  , quadrature_mode(data._quadrature_mode)
  , input_num(data._input_num)
  , input_rising(data._input_rising)
  , ticks_per_sec(data._ticks_per_sec)
{
}

hdf5pp::Type
EncoderConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EncoderConfigV2::native_type()
{
  hdf5pp::EnumType<uint32_t> countModeEnumType = hdf5pp::EnumType<uint32_t>::enumType() ;
  countModeEnumType.insert ( "WRAP_FULL", Pds::Encoder::ConfigV2::count_mode::WRAP_FULL ) ;
  countModeEnumType.insert ( "LIMIT", Pds::Encoder::ConfigV2::count_mode::LIMIT ) ;
  countModeEnumType.insert ( "HALT", Pds::Encoder::ConfigV2::count_mode::HALT ) ;
  countModeEnumType.insert ( "WRAP_PRESET", Pds::Encoder::ConfigV2::count_mode::WRAP_PRESET ) ;
  countModeEnumType.insert ( "END", Pds::Encoder::ConfigV2::count_mode::END ) ;

  hdf5pp::EnumType<uint32_t> quadModeEnumType = hdf5pp::EnumType<uint32_t>::enumType() ;
  quadModeEnumType.insert ( "CLOCK_DIR", Pds::Encoder::ConfigV2::quad_mode::CLOCK_DIR ) ;
  quadModeEnumType.insert ( "X1", Pds::Encoder::ConfigV2::quad_mode::X1 ) ;
  quadModeEnumType.insert ( "X2", Pds::Encoder::ConfigV2::quad_mode::X2 ) ;
  quadModeEnumType.insert ( "X4", Pds::Encoder::ConfigV2::quad_mode::X4 ) ;
  quadModeEnumType.insert ( "END", Pds::Encoder::ConfigV2::quad_mode::END ) ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<EncoderConfigV2>() ;
  confType.insert_native<uint32_t>( "chan_mask", offsetof(EncoderConfigV2, chan_mask) ) ;
  confType.insert( "count_mode", offsetof(EncoderConfigV2,count_mode), countModeEnumType ) ;
  confType.insert( "quadrature_mode", offsetof(EncoderConfigV2,quadrature_mode), quadModeEnumType ) ;
  confType.insert_native<uint32_t>( "input_num", offsetof(EncoderConfigV2, input_num) ) ;
  confType.insert_native<uint32_t>( "input_rising", offsetof(EncoderConfigV2, input_rising) ) ;
  confType.insert_native<uint32_t>( "ticks_per_sec", offsetof(EncoderConfigV2, ticks_per_sec) ) ;

  return confType ;
}

void
EncoderConfigV2::store( const Pds::Encoder::ConfigV2& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  EncoderConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
