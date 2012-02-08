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
#include "pdsdata/camera/FrameV1.hh"

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

class CameraFrameV1  {
public:

  typedef Pds::Camera::FrameV1 XtcType ;

  CameraFrameV1 () {}
  CameraFrameV1 ( const Pds::Camera::FrameV1& frame ) ;

  ~CameraFrameV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static hdf5pp::Type imageType( const Pds::Camera::FrameV1& frame ) ;

private:

  uint32_t width ;
  uint32_t height ;
  uint32_t depth ;
  uint32_t offset ;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_CAMERAFRAMEV1_H
