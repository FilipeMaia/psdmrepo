#ifndef PYPDSDATA_FRAMECOORD_H
#define PYPDSDATA_FRAMECOORD_H

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
#include "Python.h"

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

struct FrameCoord {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  // makes new FrameCoord object from Pds type
  static PyObject* FrameCoord_FromPds(const Pds::Camera::FrameCoord& coord);

  // standard Python stuff
  PyObject_HEAD

  Pds::Camera::FrameCoord m_coord;

};

} // namespace Camera
} // namespace pypdsdata

#endif // PYPDSDATA_FRAMECOORD_H
