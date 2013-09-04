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
 *  @brief PixCoordsCSPad2x2V2 class defines the CSPAD2x2 pixel coordinates.
 *
 *  Use the same frame like in optical measurement, but in "matrix style" geometry:
 *  X axis goes along rows (from top to bottom)
 *  Y axis goes along columns (from left to right)
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see PixCoords2x1V2, PSCalib::CSPad2x2CalibPars
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */


/**   
 *  <h1>Interface Description</h1>
 *   
 *  @li  Include and typedef
 *  @code
 *  #include "CSPadPixCoords/PixCoordsCSPad2x2V2.h"
 *  #include "PSCalib/CSPad2x2CalibPars.h"
 *  //#include "CSPadPixCoords/Image2D.h"
 *
 *  typedef CSPadPixCoords::PixCoordsCSPad2x2V2 PC2X2;
 *  typedef PSCalib::CSPad2x2CalibPars CALIB2X2;
 *  @endcode
 *
 *  @li  Instatiation \n
 *  Default constructor without any parameters makes an object
 *  with default geometry which does not account for real detector geometry. 
 *  This constructor can be used for test purpose or if relative position of two 2x1s is not important. 
 *  @code
 *         PC2X2 *pix_coords_2x2 = new PC2X2();  
 *  @endcode
 *  \n 
 *  Precise detector geometry can be accounted using pointer to the store of calibration parameters:
 *  @code
 *         PC2X2 *pix_coords_2x2 = new PC2X2(calibpars2x2);
 *  @endcode
 *  or with additional parameters:
 *  @code
 *         bool tiltIsApplied = true, 
 *         bool use_wide_pix_center=false;
 *         PC2X2 *pix_coords_2x2 = new PC2X2(calibpars2x2, tiltIsApplied, use_wide_pix_center);  
 *  @endcode
 *  where calibpars2x2 is a properly prio-initialized object of the class PSCalib/CSPad2x2CalibPars.
 *  For example, for explisit list of parameters,
 *  @code
 *         const std::string calibDir  = "/reg/d/psdm/xpp/xpptut13/calib";
 *         const std::string groupName = "CsPad2x2::CalibV1";
 *         const std::string source    = "XppGon.0:Cspad2x2.1";
 *         unsigned          runNumber = 10;
 *         CALIB2X2 *calibpars2x2 = new CALIB2X2(calibDir, groupName, source, runNumber);  
 *  @endcode
 *       
 *       
 *  @li  Printing methods
 *  @code
 *         pix_coords_2x2 -> printXYLimits();
 *         pix_coords_2x2 -> printConstants();
 *  
 *         unsigned row_begin=15, row_end=20, col_begin=40, col_end=50;
 *         pix_coords_2x2 -> printCoordArray(row_begin, row_end, col_begin, col_end);
 *  @endcode
 *
 *
 *  @li  Access methods
 *  Get pixel coordinate for specified axia, section, row, and column numbers
 *  @code
 *         unsigned s=1, r=123, c=235;
 *         double ix = pix_coords_2x2 -> getPixCoor_pix(PC2X2::AXIS_X, s, r, c);
 *         double iy = pix_coords_2x2 -> getPixCoor_pix(PC2X2::AXIS_Y, s, r, c);
 *  @endcode
 *
 *
 *  @li Example of how to compose image of pixel coordinates:
 *  @code
 *         const unsigned NX=400, NY=400;
 *         double img_arr[NY][NX];
 *         std::fill_n(&img_arr[0][0], int(NX*NY), double(0));
 *        
 *         for (unsigned r=0; r<PC2X2::ROWS2X1; r++){
 *         for (unsigned c=0; c<PC2X2::COLS2X1; c++){
 *         for (unsigned s=0; s<PC2X2::N2X1_IN_DET; s++){
 *        
 *         int ix = int (pix_coords_2x2 -> getPixCoor_pix(PC2X2::AXIS_X, s, r, c) + 0.1);
 *         int iy = int (pix_coords_2x2 -> getPixCoor_pix(PC2X2::AXIS_Y, s, r, c) + 0.1);
 *       
 *           img_arr[ix][iy] = ix+iy; // <--- This should be an intensity in this pixel.
 *         }
 *         }
 *         }
 *  @endcode
 *       
 */
 
class PixCoordsCSPad2x2V2 : public PixCoords2x1V2 {
public:
  /// Number of 2x1s in the CSPAD2x2
  const static unsigned N2X1_IN_DET = 2;
 
  // Default constructor
  //PixCoordsCSPad2x2V2 ();

  /**
   *  @brief PixCoordsCSPad2x2V2 class fills and provides access to the CSPAD2x2 pixel coordinates.
   *  
   *  @param[in] cspad_calibpars - pointer to the store of CSPAD2x2 calibration parameters
   *  @param[in] tiltIsApplied - boolean key indicating if the tilt angle correction for 2x1 in the detector is applied.
   *  @param[in] use_wide_pix_center - boolean parameter defining coordinate of the wide pixel; true-use wide pixel center as its coordinate, false-use ASIC-uniform pixel coordinate.
   */
  PixCoordsCSPad2x2V2 (PSCalib::CSPad2x2CalibPars *cspad_calibpars = new PSCalib::CSPad2x2CalibPars(), 
                       bool tiltIsApplied = true, 
                       bool use_wide_pix_center=false);

  /// Destructor
  virtual ~PixCoordsCSPad2x2V2 () ;

  /// Prints X and Y limits of the pixel coordinate map
  void printXYLimits();

  /// Prints member data and partial coordinate map
  void printConstants();

  /// Prints the part of the 2-D coordinate array in the specified ranges of rows and columns  
  void printCoordArray(unsigned r1=10, unsigned r2=21, unsigned c1=15, unsigned c2=18);

  /**
   *  @brief Returns coordimate of the pixel in [um](micrometers) for specified axis, section, row, and column
   *  @param[in] axis - enomerated axes, can be PC2X2::AXIS_X or PC2X2::AXIS_Y
   *  @param[in] sect - section index [0,1]
   *  @param[in] row - row index [0,184]
   *  @param[in] col - column index [0,387]
   */
  double getPixCoor_um (AXIS axis, unsigned sect, unsigned row, unsigned col) ;

  /**
   *  @brief Returns coordimate of the pixel in [pix](pixel size) for specified axis, section, row, and column
   *  @param[in] axis - enumerated axes, can be set to PC2X2::AXIS_X or PC2X2::AXIS_Y
   *  @param[in] sect - section index [0,1] (two 2x1 sensors in CSPAD2x2)
   *  @param[in] row - row index [0,184]
   *  @param[in] col - column index [0,387]
   */
  double getPixCoor_pix(AXIS axis, unsigned sect, unsigned row, unsigned col) ;


  /// Returns minimal x coordinate of the pixel in [um]
  double get_x_min() { return m_coor_x_min; };

  /// Returns maximal x coordinate of the pixel in [um]
  double get_x_max() { return m_coor_x_max; };

  /// Returns minimal y coordinate of the pixel in [um]
  double get_y_min() { return m_coor_y_min; };

  /// Returns maximal y coordinate of the pixel in [um]
  double get_y_max() { return m_coor_y_max; };

protected:
  /// Protected method for filling pixel coordinate array in constructor
  void fillPixelCoordinateArrays();

  /// Protected method for filling pixel coordinate array for specified sensor
  void fillOneSectionInDet(uint32_t sect, double xcenter, double ycenter, double zcenter, double rotation);

  /// Protected method which resets the origin of the pixel coordinate map to (0,0)
  void resetXYOriginAndMinMax();

private:

  /// Pointer to the store of calibration parameters
  PSCalib::CSPad2x2CalibPars *m_cspad2x2_calibpars;  

  /// Flag showing whether tilt angle needs to be applied.
  bool                        m_tiltIsApplied;

  /// Array of x pixel coordinates in [um]
  double m_coor_x[ROWS2X1][COLS2X1][N2X1_IN_DET];

  /// Array of y pixel coordinates in [um]
  double m_coor_y[ROWS2X1][COLS2X1][N2X1_IN_DET];

  /// Minimal x coordinate of the pixel in [um]
  double m_coor_x_min;

  /// Maximal x coordinate of the pixel in [um]
  double m_coor_x_max;

  /// Minimal y coordinate of the pixel in [um]
  double m_coor_y_min;

  /// Maximal y coordinate of the pixel in [um]
  double m_coor_y_max;

  /// Copy constructor
  PixCoordsCSPad2x2V2 ( const PixCoordsCSPad2x2V2& ) ;

  /// Assignment constructor
  PixCoordsCSPad2x2V2& operator = ( const PixCoordsCSPad2x2V2& ) ;
};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDSCSPAD2X2V2_H
