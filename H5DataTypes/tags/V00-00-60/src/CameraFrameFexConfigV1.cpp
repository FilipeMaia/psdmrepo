//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraFrameFexConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CameraFrameFexConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/CameraFrameCoordV1.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

CameraFrameFexConfigV1::CameraFrameFexConfigV1 (const Pds::Camera::FrameFexConfigV1& config)
{
  m_data.forwarding = config.forwarding() ;
  m_data.forward_prescale = config.forward_prescale() ;
  m_data.processing = config.processing() ;
  new(&m_data.roiBegin) CameraFrameCoordV1(config.roiBegin());
  new(&m_data.roiEnd) CameraFrameCoordV1(config.roiEnd());
  m_data.threshold = config.threshold() ;
  m_data.number_of_masked_pixels = config.number_of_masked_pixels() ;
}

hdf5pp::Type
CameraFrameFexConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CameraFrameFexConfigV1::native_type()
{
  hdf5pp::EnumType<uint32_t> forwardingEnum = hdf5pp::EnumType<uint32_t>::enumType() ;
  forwardingEnum.insert ( "NoFrame", Pds::Camera::FrameFexConfigV1::NoFrame ) ;
  forwardingEnum.insert ( "FullFrame", Pds::Camera::FrameFexConfigV1::FullFrame ) ;
  forwardingEnum.insert ( "RegionOfInterest", Pds::Camera::FrameFexConfigV1::RegionOfInterest ) ;

  hdf5pp::EnumType<uint32_t> processingEnum = hdf5pp::EnumType<uint32_t>::enumType() ;
  processingEnum.insert ( "NoProcessing", Pds::Camera::FrameFexConfigV1::NoProcessing ) ;
  processingEnum.insert ( "GssFullFrame", Pds::Camera::FrameFexConfigV1::GssFullFrame ) ;
  processingEnum.insert ( "GssRegionOfInterest", Pds::Camera::FrameFexConfigV1::GssRegionOfInterest ) ;
  processingEnum.insert ( "GssThreshold", Pds::Camera::FrameFexConfigV1::GssThreshold ) ;

  hdf5pp::Type coordType = hdf5pp::TypeTraits<CameraFrameCoordV1>::native_type() ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<CameraFrameFexConfigV1>() ;
  confType.insert( "forwarding", offsetof(CameraFrameFexConfigV1_Data,forwarding), forwardingEnum ) ;
  confType.insert_native<uint32_t>( "forward_prescale", offsetof(CameraFrameFexConfigV1_Data,forward_prescale) ) ;
  confType.insert( "processing", offsetof(CameraFrameFexConfigV1_Data,processing), processingEnum ) ;
  confType.insert( "roiBegin", offsetof(CameraFrameFexConfigV1_Data,roiBegin), coordType ) ;
  confType.insert( "roiEnd", offsetof(CameraFrameFexConfigV1_Data,roiEnd), coordType ) ;
  confType.insert_native<uint32_t>( "threshold", offsetof(CameraFrameFexConfigV1_Data,threshold) ) ;
  confType.insert_native<uint32_t>( "number_of_masked_pixels", offsetof(CameraFrameFexConfigV1_Data,number_of_masked_pixels) ) ;

  return confType ;
}

void
CameraFrameFexConfigV1::store( const Pds::Camera::FrameFexConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  CameraFrameFexConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;

  // make array data set for masked pixels
  const uint32_t maskedSize = config.number_of_masked_pixels() ;
  storeCameraFrameCoordV1 ( maskedSize, &config.masked_pixel_coordinates(), grp, "masked_pixel_coordinates" ) ;
}

} // namespace H5DataTypes
