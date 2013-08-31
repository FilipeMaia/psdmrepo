#ifndef CSPADPIXCOORDS_PIXCOORDSCSPADV2_H
#define CSPADPIXCOORDS_PIXCOORDSCSPADV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsCSPadV2.
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
#include "CSPadPixCoords/CSPadConfigPars.h"
#include "PSCalib/CSPadCalibPars.h"
#include "ndarray/ndarray.h"

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
 *  @brief PixCoordsCSPadV2 class defines the CSPad pixel coordinates in the detector.
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

class PixCoordsCSPadV2 : public PixCoords2x1V2 {
public:

  const static unsigned N2X1_IN_DET = 32;
 
  // Default constructor
  /**
   *  @brief PixCoordsCSPadV2 class fills and provides access to the CSPad pixel coordinates.
   *  
   *  Fills/holds/provides access to the array of the quad coordinates, indexed by the quad, section, row, and column.
   *  @param[in] cspad_calibpars - pointer to the store of CSPAD calibration parameters
   *  @param[in] tiltIsApplied - boolean key indicating if the tilt angle correction for 2x1 in the detector is applied.
   *  @param[in] use_wide_pix_center - boolean parameter defining coordinate of the wide pixel; true-use wide pixel center as its coordinate, false-use ASIC-uniform pixel coordinate.
   */
  PixCoordsCSPadV2 ( PSCalib::CSPadCalibPars *cspad_calibpars = new PSCalib::CSPadCalibPars(), 
                     bool tiltIsApplied = true, 
                     bool use_wide_pix_center = false );

  // Destructor
  virtual ~PixCoordsCSPadV2 () ;

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
  double getPixCoor_um    (AXIS axis, unsigned sect, unsigned row, unsigned col) ;
  double getPixCoor_pix   (AXIS axis, unsigned sect, unsigned row, unsigned col) ;

  double* getPixCoorArr_um (AXIS axis) ;

/**
  *  @param[in] axis - enumerated axis
  *  @param[in] cspad_configpars - pointer to the store of CSPAD configuration parameters
  */
  ndarray<double,3> getPixCoorNDArrShapedAsData_um(AXIS axis, CSPadConfigPars *cspad_configpars = new CSPadConfigPars());

  double get_x_min() { return m_coor_x_min; };
  double get_x_max() { return m_coor_x_max; };
  double get_y_min() { return m_coor_y_min; };
  double get_y_max() { return m_coor_y_max; };
  double get_z_min() { return m_coor_z_min; };
  double get_z_max() { return m_coor_z_max; };

protected:

private:

  PSCalib::CSPadCalibPars *m_cspad_calibpars;
  bool                     m_tiltIsApplied;

  double m_coor_x[N2X1_IN_DET][ROWS2X1][COLS2X1]; // units: um
  double m_coor_y[N2X1_IN_DET][ROWS2X1][COLS2X1]; // units: um
  double m_coor_z[N2X1_IN_DET][ROWS2X1][COLS2X1]; // units: um

  double m_coor_x_min; // units: um
  double m_coor_x_max; // units: um
  double m_coor_y_min; // units: um
  double m_coor_y_max; // units: um
  double m_coor_z_min; // units: um
  double m_coor_z_max; // units: um

  // Copy constructor and assignment are disabled by default
  PixCoordsCSPadV2 ( const PixCoordsCSPadV2& ) ;
  PixCoordsCSPadV2& operator = ( const PixCoordsCSPadV2& ) ;
};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDSCSPADV2_H
