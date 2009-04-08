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

struct CameraFrameV1_Data  {
  double xyz ;
};

class CameraFrameV1  {
public:

  CameraFrameV1 () {}
  CameraFrameV1 ( const Pds::Camera::FrameV1& frame ) ;

  ~CameraFrameV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  CameraFrameV1_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_CAMERAFRAMEV1_H
