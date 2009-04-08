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

/**
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace H5DataTypes {

struct CameraFrameFexConfigV1_Data {
  uint32_t   forwarding;
  uint32_t   forward_prescale;
  uint32_t   processing;
  CameraFrameCoordV1_Data roiBegin;
  CameraFrameCoordV1_Data roiEnd;
  uint32_t   threshold;
  uint32_t   number_of_masked_pixels;
};

class CameraFrameFexConfigV1  {
public:

  CameraFrameFexConfigV1 () {}
  CameraFrameFexConfigV1 (const Pds::Camera::FrameFexConfigV1& config) ;

  static hdf5pp::Type persType() ;

private:
  CameraFrameFexConfigV1_Data m_data ;
};

void storeCameraFrameFexConfigV1 ( const Pds::Camera::FrameFexConfigV1& config, hdf5pp::Group location ) ;

} // namespace H5DataTypes

#endif // H5DATATYPES_CAMERAFRAMEFEXCONFIGV1_H
