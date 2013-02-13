#ifndef H5DATATYPES_CAMERAFRAMECOORDV1_H
#define H5DATATYPES_CAMERAFRAMECOORDV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraFrameCoordV1.
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
#include "hdf5pp/Group.h"
#include "pdsdata/camera/FrameCoord.hh"

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

struct CameraFrameCoordV1_Data {
  uint16_t column;
  uint16_t row;
};

class CameraFrameCoordV1 {
public:
  CameraFrameCoordV1() {}
  CameraFrameCoordV1( const Pds::Camera::FrameCoord& coord ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  CameraFrameCoordV1_Data m_data ;
};

void storeCameraFrameCoordV1 ( hsize_t size, const Pds::Camera::FrameCoord* coord, hdf5pp::Group grp, const char* name ) ;

} // namespace H5DataTypes

#endif // H5DATATYPES_CAMERAFRAMECOORDV1_H
