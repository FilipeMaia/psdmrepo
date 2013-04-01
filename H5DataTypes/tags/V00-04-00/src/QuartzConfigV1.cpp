//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class QuartzConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/QuartzConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/CameraFrameCoordV1.h"
#include "H5DataTypes/H5DataUtils.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

QuartzConfigV1::QuartzConfigV1 ( const Pds::Quartz::ConfigV1& config )
  : black_level(config.black_level())
  , gain_percent(config.gain_percent())
  , output_offset(config.output_offset())
  , output_resolution(config.output_resolution())
  , output_resolution_bits(config.output_resolution_bits())
  , horizontal_binning(config.horizontal_binning())
  , vertical_binning(config.vertical_binning())
  , output_mirroring(config.output_mirroring())
  , defect_pixel_correction_enabled(config.defect_pixel_correction_enabled())
  , output_lookup_table_enabled(config.output_lookup_table_enabled())
  , number_of_defect_pixels(config.number_of_defect_pixels())
{
}

hdf5pp::Type
QuartzConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
QuartzConfigV1::native_type()
{
  hdf5pp::EnumType<uint8_t> depthEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  depthEnum.insert ( "Eight_bit", Pds::Quartz::ConfigV1::Eight_bit ) ;
  depthEnum.insert ( "Ten_bit", Pds::Quartz::ConfigV1::Ten_bit ) ;

  hdf5pp::EnumType<uint8_t> binningEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  binningEnum.insert ( "x1", Pds::Quartz::ConfigV1::x1 ) ;
  binningEnum.insert ( "x2", Pds::Quartz::ConfigV1::x2 ) ;
  binningEnum.insert ( "x4", Pds::Quartz::ConfigV1::x4 ) ;

  hdf5pp::EnumType<uint8_t> mirroringEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  mirroringEnum.insert ( "None", Pds::Quartz::ConfigV1::None ) ;
  mirroringEnum.insert ( "HFlip", Pds::Quartz::ConfigV1::HFlip ) ;
  mirroringEnum.insert ( "VFlip", Pds::Quartz::ConfigV1::VFlip ) ;
  mirroringEnum.insert ( "HVFlip", Pds::Quartz::ConfigV1::HVFlip ) ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<QuartzConfigV1>() ;
  confType.insert_native<uint16_t>( "black_level", offsetof(QuartzConfigV1,black_level) ) ;
  confType.insert_native<uint16_t>( "gain_percent", offsetof(QuartzConfigV1,gain_percent) ) ;
  confType.insert_native<uint16_t>( "output_offset", offsetof(QuartzConfigV1,output_offset) ) ;
  confType.insert( "output_resolution", offsetof(QuartzConfigV1,output_resolution), depthEnum ) ;
  confType.insert_native<uint8_t>( "output_resolution_bits", offsetof(QuartzConfigV1,output_resolution_bits) ) ;
  confType.insert( "horizontal_binning", offsetof(QuartzConfigV1,horizontal_binning), binningEnum ) ;
  confType.insert( "vertical_binning", offsetof(QuartzConfigV1,vertical_binning), binningEnum ) ;
  confType.insert( "output_mirroring", offsetof(QuartzConfigV1,output_mirroring), mirroringEnum ) ;
  confType.insert_native<uint8_t>( "defect_pixel_correction_enabled", offsetof(QuartzConfigV1,defect_pixel_correction_enabled) ) ;
  confType.insert_native<uint8_t>( "output_lookup_table_enabled", offsetof(QuartzConfigV1,output_lookup_table_enabled) ) ;
  confType.insert_native<uint32_t>( "number_of_defect_pixels", offsetof(QuartzConfigV1,number_of_defect_pixels) ) ;

  return confType ;
}

void
QuartzConfigV1::store( const Pds::Quartz::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  QuartzConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // make array data set for LUT
  const uint32_t lutSize = Pds::Quartz::ConfigV1::LUT_Size ;
  storeDataObjects ( lutSize, config.output_lookup_table(), "output_lookup_table", grp ) ;

  // make array data set for defect pixels
  const uint32_t defectSize = config.number_of_defect_pixels() ;
  storeCameraFrameCoordV1 ( defectSize, config.defect_pixel_coordinates(), grp, "defect_pixel_coordinates" ) ;
}

} // namespace H5DataTypes
