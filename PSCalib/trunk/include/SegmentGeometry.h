#ifndef PSCALIB_SEGMENTGEOMETRY_H
#define PSCALIB_SEGMENTGEOMETRY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// $Revision$
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
//#include <iostream>
//#include <string>
//#include <vector>
//#include <map>
//#include <fstream>  // open, close etc.
//#include <stdint.h> // for uint8_t, uint16_t etc.

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "ndarray/ndarray.h"

//-----------------------------

namespace PSCalib {

/// @addtogroup PSCalib PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief Abstract base class SegmentGeometry defines the interface to access segment pixel coordinates.
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CalibFileFinder
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @see CalibFileFinder
 *
 */

//-----------------------------

class SegmentGeometry  {
public:

  /// Enumerator for X, Y, and Z axes
  enum AXIS { AXIS_X=0,
              AXIS_Y,
              AXIS_Z };

  /// Enumerator for units of pixel coordinates expressed in micrometers or number of pixels
  enum UNITS { UM=0,
               PIX };
 
  typedef double pixel_coord_t;

  // Destructor
  virtual ~SegmentGeometry () {}

  /// Returns segment size - total number of pixels in segment
  virtual const size_t size() = 0;

  /// Returns number of rows in segment
  virtual const size_t rows() = 0;

  /// Returns number of cols in segment
  virtual const size_t cols() = 0;

  /// Returns shape of the segment {rows, cols}
  virtual const unsigned* shape() = 0;

  /// Returns pointer to the array of segment pixel coordinates for AXIS and UNITS
  virtual const pixel_coord_t* pixel_coords(AXIS axis, UNITS units) = 0;

  /// Returns ndarray of segment pixel coordinates for UNITS; 3 dimensions corresponds to X, Y, and Z, respectively
  virtual const ndarray<pixel_coord_t, 3> ndarray_of_pixel_coords(UNITS units) = 0; 

  /// Returns minimal value in the array of segment pixel coordinates for AXIS and UNITS
  virtual const pixel_coord_t pixel_coord_min(AXIS axis, UNITS units) = 0;

  /// Returns maximal value in the array of segment pixel coordinates for AXIS and UNITS
  virtual const pixel_coord_t pixel_coord_max(AXIS axis, UNITS units) = 0;
};

} // namespace PSCalib

#endif // PSCALIB_SEGMENTGEOMETRY_H

//-----------------------------
