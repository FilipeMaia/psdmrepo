#ifndef H5DATATYPES_CAMERAFRAMEV1_H
#define H5DATATYPES_CAMERAFRAMEV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraFrameV1.
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
#include "hdf5pp/Type.h"
#include "pdsdata/psddl/camera.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
//  Helper class for Pds::Camera::FrameV1 class
//
class CameraFrameV1  {
public:

  typedef Pds::Camera::FrameV1 XtcType ;

  CameraFrameV1 () {}
  CameraFrameV1 ( const Pds::Camera::FrameV1& frame ) ;

  ~CameraFrameV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& frame ) {
    size_t size = sizeof frame + frame.width()*frame.height()*((frame.depth()+7)/8);
    return ((size + 3) / 4) * 4 ;
  }

  static hdf5pp::Type imageType( const XtcType& frame ) ;

private:

  uint32_t width ;
  uint32_t height ;
  uint32_t depth ;
  uint32_t offset ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CAMERAFRAMEV1_H
