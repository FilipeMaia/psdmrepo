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
#include "pdsdata/psddl/camera.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Camera {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
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

  // dump to a stream
  void print(std::ostream& out) const;

};

} // namespace Camera
} // namespace pypdsdata

#endif // PYPDSDATA_CAMERA_TWODGAUSSIANV1_H
