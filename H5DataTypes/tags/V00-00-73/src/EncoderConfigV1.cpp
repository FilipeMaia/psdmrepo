//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EncoderConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EncoderConfigV1.h"

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

EncoderConfigV1::EncoderConfigV1 ( const Pds::Encoder::ConfigV1& data )
  : chan_num(data._chan_num)
  , count_mode(data._count_mode)
  , quadrature_mode(data._quadrature_mode)
  , input_num(data._input_num)
  , input_rising(data._input_rising)
  , ticks_per_sec(data._ticks_per_sec)
{
}

hdf5pp::Type
EncoderConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EncoderConfigV1::native_type()
{
  hdf5pp::EnumType<uint32_t> countModeEnumType = hdf5pp::EnumType<uint32_t>::enumType() ;
  countModeEnumType.insert ( "WRAP_FULL", Pds::Encoder::ConfigV1::count_mode::WRAP_FULL ) ;
  countModeEnumType.insert ( "LIMIT", Pds::Encoder::ConfigV1::count_mode::LIMIT ) ;
  countModeEnumType.insert ( "HALT", Pds::Encoder::ConfigV1::count_mode::HALT ) ;
  countModeEnumType.insert ( "WRAP_PRESET", Pds::Encoder::ConfigV1::count_mode::WRAP_PRESET ) ;
  countModeEnumType.insert ( "END", Pds::Encoder::ConfigV1::count_mode::END ) ;

  hdf5pp::EnumType<uint32_t> quadModeEnumType = hdf5pp::EnumType<uint32_t>::enumType() ;
  quadModeEnumType.insert ( "CLOCK_DIR", Pds::Encoder::ConfigV1::quad_mode::CLOCK_DIR ) ;
  quadModeEnumType.insert ( "X1", Pds::Encoder::ConfigV1::quad_mode::X1 ) ;
  quadModeEnumType.insert ( "X2", Pds::Encoder::ConfigV1::quad_mode::X2 ) ;
  quadModeEnumType.insert ( "X4", Pds::Encoder::ConfigV1::quad_mode::X4 ) ;
  quadModeEnumType.insert ( "END", Pds::Encoder::ConfigV1::quad_mode::END ) ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<EncoderConfigV1>() ;
  confType.insert_native<uint32_t>( "chan_num", offsetof(EncoderConfigV1, chan_num) ) ;
  confType.insert( "count_mode", offsetof(EncoderConfigV1,count_mode), countModeEnumType ) ;
  confType.insert( "quadrature_mode", offsetof(EncoderConfigV1,quadrature_mode), quadModeEnumType ) ;
  confType.insert_native<uint32_t>( "input_num", offsetof(EncoderConfigV1, input_num) ) ;
  confType.insert_native<uint32_t>( "input_rising", offsetof(EncoderConfigV1, input_rising) ) ;
  confType.insert_native<uint32_t>( "ticks_per_sec", offsetof(EncoderConfigV1, ticks_per_sec) ) ;

  return confType ;
}

void
EncoderConfigV1::store( const Pds::Encoder::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  EncoderConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
