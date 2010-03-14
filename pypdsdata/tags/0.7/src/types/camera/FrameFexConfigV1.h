#ifndef PYPDSDATA_CAMERA_FRAMEFEXCONFIGV1_H
#define PYPDSDATA_CAMERA_FRAMEFEXCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FrameFexConfigV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "types/PdsDataType.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/camera/FrameFexConfigV1.hh"

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

class FrameFexConfigV1 : public PdsDataType<FrameFexConfigV1,Pds::Camera::FrameFexConfigV1> {
public:

  typedef PdsDataType<FrameFexConfigV1,Pds::Camera::FrameFexConfigV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace Camera
} // namespace pypdsdata

#endif // PYPDSDATA_CAMERA_FRAMEFEXCONFIGV1_H
