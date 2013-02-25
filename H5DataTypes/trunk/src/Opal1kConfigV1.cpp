//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Opal1kConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/Opal1kConfigV1.h"

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

Opal1kConfigV1::Opal1kConfigV1 ( const Pds::Opal1k::ConfigV1& config )
  : black_level (config.black_level())
  , gain_percent (config.gain_percent())
  , output_offset (config.output_offset())
  , output_resolution (config.output_resolution())
  , output_resolution_bits (config.output_resolution_bits())
  , vertical_binning (config.vertical_binning())
  , output_mirroring (config.output_mirroring())
  , vertical_remapping (config.vertical_remapping())
  , defect_pixel_correction_enabled (config.defect_pixel_correction_enabled())
  , output_lookup_table_enabled (config.output_lookup_table_enabled())
  , number_of_defect_pixels (config.number_of_defect_pixels())
{
}

hdf5pp::Type
Opal1kConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
Opal1kConfigV1::native_type()
{
  hdf5pp::EnumType<uint8_t> depthEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  depthEnum.insert ( "Eight_bit", Pds::Opal1k::ConfigV1::Eight_bit ) ;
  depthEnum.insert ( "Ten_bit", Pds::Opal1k::ConfigV1::Ten_bit ) ;
  depthEnum.insert ( "Twelve_bit", Pds::Opal1k::ConfigV1::Twelve_bit ) ;

  hdf5pp::EnumType<uint8_t> binningEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  binningEnum.insert ( "x1", Pds::Opal1k::ConfigV1::x1 ) ;
  binningEnum.insert ( "x2", Pds::Opal1k::ConfigV1::x2 ) ;
  binningEnum.insert ( "x4", Pds::Opal1k::ConfigV1::x4 ) ;
  binningEnum.insert ( "x8", Pds::Opal1k::ConfigV1::x8 ) ;

  hdf5pp::EnumType<uint8_t> mirroringEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  mirroringEnum.insert ( "None", Pds::Opal1k::ConfigV1::None ) ;
  mirroringEnum.insert ( "HFlip", Pds::Opal1k::ConfigV1::HFlip ) ;
  mirroringEnum.insert ( "VFlip", Pds::Opal1k::ConfigV1::VFlip ) ;
  mirroringEnum.insert ( "HVFlip", Pds::Opal1k::ConfigV1::HVFlip ) ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<Opal1kConfigV1>() ;
  confType.insert_native<uint16_t>( "black_level", offsetof(Opal1kConfigV1,black_level) ) ;
  confType.insert_native<uint16_t>( "gain_percent", offsetof(Opal1kConfigV1,gain_percent) ) ;
  confType.insert_native<uint16_t>( "output_offset", offsetof(Opal1kConfigV1,output_offset) ) ;
  confType.insert( "output_resolution", offsetof(Opal1kConfigV1,output_resolution), depthEnum ) ;
  confType.insert_native<uint8_t>( "output_resolution_bits", offsetof(Opal1kConfigV1,output_resolution_bits) ) ;
  confType.insert( "vertical_binning", offsetof(Opal1kConfigV1,vertical_binning), binningEnum ) ;
  confType.insert( "output_mirroring", offsetof(Opal1kConfigV1,output_mirroring), mirroringEnum ) ;
  confType.insert_native<uint8_t>( "vertical_remapping", offsetof(Opal1kConfigV1,vertical_remapping) ) ;
  confType.insert_native<uint8_t>( "defect_pixel_correction_enabled", offsetof(Opal1kConfigV1,defect_pixel_correction_enabled) ) ;
  confType.insert_native<uint8_t>( "output_lookup_table_enabled", offsetof(Opal1kConfigV1,output_lookup_table_enabled) ) ;
  confType.insert_native<uint32_t>( "number_of_defect_pixels", offsetof(Opal1kConfigV1,number_of_defect_pixels) ) ;

  return confType ;
}

void
Opal1kConfigV1::store( const Pds::Opal1k::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  Opal1kConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // make array data set for LUT
  if (config.output_lookup_table_enabled()) {
    const uint32_t lutSize = Pds::Opal1k::ConfigV1::LUT_Size ;
    storeDataObjects ( lutSize, config.output_lookup_table(), "output_lookup_table", grp ) ;
  }
  
  // make array data set for defect pixels
  const uint32_t defectSize = config.number_of_defect_pixels() ;
  storeCameraFrameCoordV1 ( defectSize, config.defect_pixel_coordinates(), grp, "defect_pixel_coordinates" ) ;
}

} // namespace H5DataTypes
