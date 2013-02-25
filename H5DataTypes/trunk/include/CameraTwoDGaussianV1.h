#ifndef H5DATATYPES_CAMERATWODGAUSSIANV1_H
#define H5DATATYPES_CAMERATWODGAUSSIANV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraTwoDGaussianV1.
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
#include "pdsdata/camera/TwoDGaussianV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
//  Helper class for Pds::Camera::TwoDGaussianV1 class
//
class CameraTwoDGaussianV1  {
public:

  typedef Pds::Camera::TwoDGaussianV1 XtcType ;

  CameraTwoDGaussianV1 () {}
  CameraTwoDGaussianV1 ( const Pds::Camera::TwoDGaussianV1& config ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:
  uint64_t integral;
  double xmean;
  double ymean;
  double major_axis_width;
  double minor_axis_width;
  double major_axis_tilt;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_CAMERATWODGAUSSIANV1_H
