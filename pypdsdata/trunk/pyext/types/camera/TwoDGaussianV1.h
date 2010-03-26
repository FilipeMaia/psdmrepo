#ifndef PYPDSDATA_CAMERA_TWODGAUSSIANV1_H
#define PYPDSDATA_CAMERA_TWODGAUSSIANV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Camera_TwoDGaussianV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "../PdsDataType.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/camera/TwoDGaussianV1.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Camera {

/**
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class TwoDGaussianV1 : public PdsDataType<TwoDGaussianV1,Pds::Camera::TwoDGaussianV1> {
public:

  typedef PdsDataType<TwoDGaussianV1,Pds::Camera::TwoDGaussianV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace Camera
} // namespace pypdsdata

#endif // PYPDSDATA_CAMERA_TWODGAUSSIANV1_H
