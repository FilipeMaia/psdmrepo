#ifndef H5DATATYPES_CAMERAFRAMEFEXCONFIGV1_H
#define H5DATATYPES_CAMERAFRAMEFEXCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraFrameFexConfigV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/CameraFrameCoordV1.h"
#include "hdf5pp/Group.h"
#include "pdsdata/camera/FrameFexConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
//  Helper class for Pds::Camera::FrameFexConfigV1 class
//
class CameraFrameFexConfigV1  {
public:

  typedef Pds::Camera::FrameFexConfigV1 XtcType ;

  CameraFrameFexConfigV1 () {}
  CameraFrameFexConfigV1 (const Pds::Camera::FrameFexConfigV1& config) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const Pds::Camera::FrameFexConfigV1& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc.size() ; }

private:
  uint32_t   forwarding;
  uint32_t   forward_prescale;
  uint32_t   processing;
  CameraFrameCoordV1 roiBegin;
  CameraFrameCoordV1 roiEnd;
  uint32_t   threshold;
  uint32_t   number_of_masked_pixels;
};


} // namespace H5DataTypes

#endif // H5DATATYPES_CAMERAFRAMEFEXCONFIGV1_H
