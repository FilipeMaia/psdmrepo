#ifndef CSPADPIXCOORDS_PIXCOORDSCSPAD2X2V2_H
#define CSPADPIXCOORDS_PIXCOORDSCSPAD2X2V2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsCSPad2x2V2.
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
#include "CSPadPixCoords/PixCoords2x1V2.h"
#include "PSCalib/CSPad2x2CalibPars.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief PixCoordsCSPad2x2V2 class defines the CSPad2x2 pixel coordinates in its local frame.
 *
 *  Use the same frame like in optical measurement, but in "matrix style" geometry:
 *  X axis goes along rows (from top to bottom)
 *  Y axis goes along columns (from left to right)
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CSPadImageProducer, PixCoordsTest
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class PixCoordsCSPad2x2V2 : public PixCoords2x1V2 {
public:

  const static unsigned N2X1_IN_DET = 2;
 
  // Default constructor
  /**
   *  @brief PixCoordsCSPad2x2V2 class fills and provides access to the CSPad2x2 pixel coordinates.
   *  
   *  Fills/holds/provides access to the array of the quad coordinates, indexed by the quad, section, row, and column.
   *  @param[in] cspad_calibpars - pointer to the store of CSPAD2x2 calibration parameters
   *  @param[in] tiltIsApplied - boolean key indicating if the tilt angle correction for 2x1 in the detector is applied.
   *  @param[in] use_wide_pix_center - boolean parameter defining coordinate of the wide pixel; true-use wide pixel center as its coordinate, false-use ASIC-uniform pixel coordinate.
   */

  //PixCoordsCSPad2x2V2 ();

  PixCoordsCSPad2x2V2 (PSCalib::CSPad2x2CalibPars *cspad_calibpars = new PSCalib::CSPad2x2CalibPars(), 
                       bool tiltIsApplied = true, 
                       bool use_wide_pix_center=false);

  // Destructor
  virtual ~PixCoordsCSPad2x2V2 () ;

  void fillPixelCoordinateArrays();
  void fillOneSectionInDet(uint32_t sect, double xcenter, double ycenter, double zcenter, double rotation);
  void resetXYOriginAndMinMax();
  void printXYLimits();
  void printConstants();
  void printCoordArray(unsigned r1=10, unsigned r2=21, unsigned c1=15, unsigned c2=18);

  /**
   *  Access methods return the coordinate for indicated axis, quad, section, row, and column
   *  indexes after the quad rotation by n*90 degree.
   *  The pixel coordinates can be returned in um(micrometers) and pix(pixels).
   */
  double getPixCoor_um (AXIS axis, unsigned sect, unsigned row, unsigned col) ;
  double getPixCoor_pix(AXIS axis, unsigned sect, unsigned row, unsigned col) ;
  double get_x_min() { return m_coor_x_min; }; // units: um
  double get_x_max() { return m_coor_x_max; }; // units: um
  double get_y_min() { return m_coor_y_min; }; // units: um
  double get_y_max() { return m_coor_y_max; }; // units: um

protected:

private:

  PSCalib::CSPad2x2CalibPars *m_cspad2x2_calibpars;  
  bool                        m_tiltIsApplied;

  double m_coor_x[ROWS2X1][COLS2X1][N2X1_IN_DET]; // units: um
  double m_coor_y[ROWS2X1][COLS2X1][N2X1_IN_DET]; // units: um

  double m_coor_x_min; // units: um
  double m_coor_x_max; // units: um
  double m_coor_y_min; // units: um
  double m_coor_y_max; // units: um

  // Copy constructor and assignment are disabled by default
  PixCoordsCSPad2x2V2 ( const PixCoordsCSPad2x2V2& ) ;
  PixCoordsCSPad2x2V2& operator = ( const PixCoordsCSPad2x2V2& ) ;
};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDSCSPAD2X2V2_H
