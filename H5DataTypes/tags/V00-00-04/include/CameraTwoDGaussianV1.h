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
#include "hdf5pp/DataSet.h"
#include "pdsdata/camera/TwoDGaussianV1.hh"

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

struct CameraTwoDGaussianV1_Data {
  uint64_t integral;
  double xmean;
  double ymean;
  double major_axis_width;
  double minor_axis_width;
  double major_axis_tilt;
};

class CameraTwoDGaussianV1  {
public:

  typedef Pds::Camera::TwoDGaussianV1 XtcType ;

  CameraTwoDGaussianV1 () {}
  CameraTwoDGaussianV1 ( const Pds::Camera::TwoDGaussianV1& config ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  CameraTwoDGaussianV1_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_CAMERATWODGAUSSIANV1_H
