#ifndef PYPDSDATA_CAMERA_FRAMECOORD_H
#define PYPDSDATA_CAMERA_FRAMECOORD_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FrameCoord.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "../PdsDataTypeEmbedded.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/camera/FrameCoord.hh"

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

class FrameCoord : public PdsDataTypeEmbedded<FrameCoord,Pds::Camera::FrameCoord> {
public:

  typedef PdsDataTypeEmbedded<FrameCoord,Pds::Camera::FrameCoord> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace Camera
} // namespace pypdsdata

#endif // PYPDSDATA_CAMERA_FRAMECOORD_H
