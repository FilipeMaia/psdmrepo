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
#include "Lusi/Lusi.h"

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

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

Opal1kConfigV1::Opal1kConfigV1 ( const Pds::Opal1k::ConfigV1& config )
{
  m_data.black_level = config.black_level() ;
  m_data.gain_percent = config.gain_percent() ;
  m_data.output_offset = config.output_offset() ;
  m_data.output_resolution = config.output_resolution() ;
  m_data.output_resolution_bits = config.output_resolution_bits() ;
  m_data.vertical_binning = config.vertical_binning() ;
  m_data.output_mirroring = config.output_mirroring() ;
  m_data.vertical_remapping = config.vertical_remapping() ;
  m_data.defect_pixel_correction_enabled = config.defect_pixel_correction_enabled() ;
  m_data.output_lookup_table_enabled = config.output_lookup_table_enabled() ;
  m_data.number_of_defect_pixels = config.number_of_defect_pixels() ;
}

hdf5pp::Type
Opal1kConfigV1::persType()
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
  confType.insert( "black_level", offsetof(Opal1kConfigV1_Data,black_level), hdf5pp::AtomicType::atomicType<uint16_t>() ) ;
  confType.insert( "gain_percent", offsetof(Opal1kConfigV1_Data,gain_percent), hdf5pp::AtomicType::atomicType<uint16_t>() ) ;
  confType.insert( "output_offset", offsetof(Opal1kConfigV1_Data,output_offset), hdf5pp::AtomicType::atomicType<uint16_t>() ) ;
  confType.insert( "output_resolution", offsetof(Opal1kConfigV1_Data,output_resolution), depthEnum ) ;
  confType.insert( "output_resolution_bits", offsetof(Opal1kConfigV1_Data,output_resolution_bits), hdf5pp::AtomicType::atomicType<uint8_t>() ) ;
  confType.insert( "vertical_binning", offsetof(Opal1kConfigV1_Data,vertical_binning), binningEnum ) ;
  confType.insert( "output_mirroring", offsetof(Opal1kConfigV1_Data,output_mirroring), mirroringEnum ) ;
  confType.insert( "vertical_remapping", offsetof(Opal1kConfigV1_Data,vertical_remapping), hdf5pp::AtomicType::atomicType<uint8_t>() ) ;
  confType.insert( "defect_pixel_correction_enabled", offsetof(Opal1kConfigV1_Data,defect_pixel_correction_enabled), hdf5pp::AtomicType::atomicType<uint8_t>() ) ;
  confType.insert( "output_lookup_table_enabled", offsetof(Opal1kConfigV1_Data,output_lookup_table_enabled), hdf5pp::AtomicType::atomicType<uint8_t>() ) ;
  confType.insert( "number_of_defect_pixels", offsetof(Opal1kConfigV1_Data,number_of_defect_pixels), hdf5pp::AtomicType::atomicType<uint32_t>() ) ;

  return confType ;
}

void
storeOpal1kConfigV1 ( const Pds::Opal1k::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  Opal1kConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // make array data set for LUT
  const uint32_t lutSize = Pds::Opal1k::ConfigV1::LUT_Size ;
  hdf5pp::Type lutType = hdf5pp::AtomicType::atomicType<uint16_t>() ;
  storeDataObjects ( lutSize, config.output_lookup_table(), lutType, "output_lookup_table", grp ) ;

  // make array data set for defect pixels
  const uint32_t defectSize = config.number_of_defect_pixels() ;
  storeCameraFrameCoordV1 ( defectSize, config.defect_pixel_coordinates(), grp, "defect_pixel_coordinates" ) ;
}

} // namespace H5DataTypes
