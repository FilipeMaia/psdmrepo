#ifndef CSPADPIXCOORDS_PIXCOORDSCSPAD2X2_H
#define CSPADPIXCOORDS_PIXCOORDSCSPAD2X2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsCSPad2x2.
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
#include "CSPadPixCoords/PixCoords2x1.h"
#include "PSCalib/CSPadCalibPars.h"

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
 *  @brief PixCoordsCSPad2x2 class defines the CSPad2x2 pixel coordinates in its local frame.
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

class PixCoordsCSPad2x2  {
public:

  enum { N2x1InDet = 2 }; 
  enum { NCols2x1  = Psana::CsPad::ColumnsPerASIC     }; // 185
  enum { NRows2x1  = Psana::CsPad::MaxRowsPerASIC * 2 }; // 194*2 = 388

  // Default constructor
  /**
   *  @brief PixCoordsCSPad2x2 class fills and provides access to the CSPad2x2 pixel coordinates.
   *  
   *  Fills/holds/provides access to the array of the quad coordinates, indexed by the quad, section, row, and column.
   *  @param[in] pix_coords_2x1  Pointer to the object with 2x1 section pixel coordinates.
   *  @param[in] tiltIsApplied   Boolean key indicating if the tilt angle correction for 2x1 in 2x2 is applied.
   *             Currently is not used, because tilts for 2x1 in 2x2 are not presented in calibtration parameters. 
   */
  PixCoordsCSPad2x2 (PixCoords2x1 *pix_coords_2x1, bool tiltIsApplied = false);
  PixCoordsCSPad2x2 (PixCoords2x1 *pix_coords_2x1, PSCalib::CSPadCalibPars *cspad_calibpar, bool tiltIsApplied = false);

  // Destructor
  virtual ~PixCoordsCSPad2x2 () ;

  void fillPixelCoordinateArrays();

  void fillOneSectionInDet      (uint32_t sect, double xcenter, double ycenter, double zcenter, double rotation);
  void fillOneSectionTiltedInDet(uint32_t sect, double xcenter, double ycenter, double zcenter, double rotation, double tilt);
  void setConstXYMinMax();

  /**
   *  Access methods return the coordinate for indicated axis, quad, section, row, and column
   *  indexes after the quad rotation by n*90 degree.
   *  The pixel coordinates can be returned in um(micrometers) and pix(pixels).
   */
  double getPixCoor_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned sect, unsigned row, unsigned col) ;
  double getPixCoor_pix(CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned sect, unsigned row, unsigned col) ;

protected:

private:

  // Data members
  CSPadPixCoords::PixCoords2x1::COORDINATE XCOOR;
  CSPadPixCoords::PixCoords2x1::COORDINATE YCOOR;
  CSPadPixCoords::PixCoords2x1::COORDINATE ZCOOR;

  double m_degToRad; 

  PixCoords2x1            *m_pix_coords_2x1;  
  PSCalib::CSPadCalibPars *m_cspad_calibpar;  
  bool                     m_tiltIsApplied;

  double m_coor_x[NCols2x1][NRows2x1][N2x1InDet];
  double m_coor_y[NCols2x1][NRows2x1][N2x1InDet];

  double m_coor_x_min;
  double m_coor_x_max;
  double m_coor_y_min;
  double m_coor_y_max;

  // Copy constructor and assignment are disabled by default
  PixCoordsCSPad2x2 ( const PixCoordsCSPad2x2& ) ;
  PixCoordsCSPad2x2& operator = ( const PixCoordsCSPad2x2& ) ;
};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDSCSPAD2X2_H
