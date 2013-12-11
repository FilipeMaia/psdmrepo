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



/**   
 *  <h1>Interface Description</h1>
 *   
 *  @li  Includes and typedefs
 *  @code
 *  #include "PSCalib/CSPadCalibPars.h"
 *  #include "CSPadPixCoords/CSPadConfigPars.h"
 *  #include "CSPadPixCoords/PixCoordsCSPadV2.h"
 *
 *  typedef PSCalib::CSPadCalibPars CALIB;
 *  typedef CSPadPixCoords::CSPadConfigPars CONFIG;
 *  typedef CSPadPixCoords::PixCoordsCSPadV2 PC;
 *  @endcode
 *  
 *  @li  Instatiation\n
 *  Default constructor does not provide correct CSPAD geometry, but can be used for test or when precise geometry is not important:
 *  @code
 *        PC *pix_coords = new PC();  
 *  @endcode
 *  Precise geometry can be obtained using pointer to the store of calibration parameters:
 *  @code
 *        PC *pix_coords = new PC(calibpars);  
 *  @endcode
 *  Instatiation of calibration parameters can be done by different ways. For example:
 *  @code
 *        const std::string calibDir  = "/reg/d/psdm/cxi/cxitut13/calib";
 *        const std::string groupName = "CsPad::CalibV1";
 *        const std::string source    = "CxiDs1.0:Cspad.0";
 *        unsigned          runNumber = 10;
 *        CALIB *calibpars = new CALIB(calibDir, groupName, source, runNumber);  
 *        calibpars->printCalibPars();
 *  @endcode
 *  
 *  @li  Printing methods\n
 *  @code
 *        pix_coords -> printXYLimits();
 *        pix_coords -> printConstants();
 *        unsigned row_begin=15, row_end=20, col_begin=40, col_end=50;
 *        pix_coords -> printCoordArray(row_begin, row_end, col_begin, col_end);
 *  @endcode
 * 
 *  @li  Access methods\n
 *  Get pixel coordinate for specified axis, section, row, and column numbers
 *  @code
 *         unsigned s=23, r=123, c=235;
 *         double ix = pix_coords -> getPixCoor_pix(PC::AXIS_X, s, r, c);
 *         double iy = pix_coords -> getPixCoor_pix(PC::AXIS_Y, s, r, c);
 *  @endcode
 *  \n
 *  Example of how to compose image of pixel coordinates:\n
 *  @code
 *        // Reservation of memory for image array
 *        unsigned NX = (unsigned)(pix_coords -> get_x_max() * PC::UM_TO_PIX + 1); 
 *        unsigned NY = (unsigned)(pix_coords -> get_x_max() * PC::UM_TO_PIX + 1);   
 *        double* img_arr = new double[NX*NY];
 *        std::fill_n(img_arr, int(NX*NY), double(0));
 *
 *        // Assignment to coordinates
 *        for (unsigned s=0; s<PC::N2X1_IN_DET; s++){
 *        for (unsigned r=0; r<PC::ROWS2X1; r++){
 *        for (unsigned c=0; c<PC::COLS2X1; c++){
 *
 *          int ix = int (pix_coords -> getPixCoor_pix(PC::AXIS_X, s, r, c) + 0.1);
 *          int iy = int (pix_coords -> getPixCoor_pix(PC::AXIS_Y, s, r, c) + 0.1);
 *
 *          img_arr[ix + iy*NX] = r+c;
 *        }
 *        }
 *        }
 *  @endcode
 *  \n
 *  Access to ndarray of pixel coordinates, taking into account CSPAD configuration parameters:
 *  @code
 *        ndarray<double,3> nda_pix_coord_x = pix_coords -> getPixCoorNDArrShapedAsData_um (PC::AXIS_X, config);
 *        ndarray<double,3> nda_pix_coord_y = pix_coords -> getPixCoorNDArrShapedAsData_um (PC::AXIS_Y, config);
 *  @endcode
 *  where the configuration parameters can be defined by different ways, for example:
 *  @code
 *        uint32_t numQuads         = 4;                     // 4; 
 *        uint32_t quadNumber[]     = {0,1,2,3};             // {0,1,2,3};
 *        uint32_t roiMask[]        = {0375,0337,0177,0376}; // {0377,0377,0377,0377};
 *        CONFIG *config = new CONFIG( numQuads, quadNumber, roiMask );  
 *        config -> printCSPadConfigPars();
 *  @endcode
 *  \n     
 *  Example of how to compose image of pixel coordinates using ndarray:
 *  @code
 *        // Assignment to coordinates for entire array
 *        int ix, iy;
 *        ndarray<double, 3>::iterator xit;
 *        ndarray<double, 3>::iterator yit;
 *        for(xit=nda_pix_coord_x.begin(), yit=nda_pix_coord_y.begin(); xit!=nda_pix_coord_x.end(); ++xit, ++yit) { 
 *          ix = int ( *xit * PC::UM_TO_PIX + 0.1);
 *          iy = int ( *yit * PC::UM_TO_PIX + 0.1);
 *          img_arr[ix + iy*NX] = ix+iy;
 *        }
 *  @endcode
 *  
 */


class PixCoordsCSPadV2 : public PixCoords2x1V2 {
public:
  /// Number of 2x1s in the entire detector
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

  /// Destructor
  virtual ~PixCoordsCSPadV2 () ;

  /// Protected method for filling pixel coordinate array in constructor
  void fillPixelCoordinateArrays();

  /// Protected method for filling pixel coordinate array for specified sensor
  void fillOneSectionInDet(uint32_t sect, double xcenter, double ycenter, double zcenter, double rotation);

  /// Protected method which resets the origin of the pixel coordinate map to (0,0)
  void resetXYOriginAndMinMax();

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
  double getPixCoor_um    (AXIS axis, unsigned sect, unsigned row, unsigned col) ;


  /**
   *  @brief Returns coordimate of the pixel in [pix](pixel size) for specified axis, section, row, and column
   *  @param[in] axis - enumerated axes, can be set to PC2X2::AXIS_X or PC2X2::AXIS_Y
   *  @param[in] sect - section index [0,1] (two 2x1 sensors in CSPAD2x2)
   *  @param[in] row - row index [0,184]
   *  @param[in] col - column index [0,387]
   */
  double getPixCoor_pix   (AXIS axis, unsigned sect, unsigned row, unsigned col) ;


  /**
   *  @brief Returns pointer to the pixel coordimate array of entire CSPAD in [um](micrometers)
   */
  double* getPixCoorArr_um (AXIS axis) ;

/** Returns the ndarray with data for specified axis and configuration parameters
  *  @param[in] axis - enumerated axis
  *  @param[in] cspad_configpars - pointer to the store of CSPAD configuration parameters
  */
  ndarray<double,3> getPixCoorNDArrShapedAsData_um(AXIS axis, CSPadConfigPars *cspad_configpars = new CSPadConfigPars());

  /// Returns minimal x coordinate of the pixel in [um]
  double get_x_min() { return m_coor_x_min; };

  /// Returns maximal x coordinate of the pixel in [um]
  double get_x_max() { return m_coor_x_max; };

  /// Returns minimal y coordinate of the pixel in [um]
  double get_y_min() { return m_coor_y_min; };

  /// Returns maximal y coordinate of the pixel in [um]
  double get_y_max() { return m_coor_y_max; };

  /// Returns minimal z coordinate of the pixel in [um]
  double get_z_min() { return m_coor_z_min; };

  /// Returns maximal z coordinate of the pixel in [um]
  double get_z_max() { return m_coor_z_max; };

protected:

private:
  /// Pointer to the store of calibration parameters
  PSCalib::CSPadCalibPars *m_cspad_calibpars;

  /// Flag showing whether tilt angle needs to be applied.
  bool                     m_tiltIsApplied;

  /// Array of x pixel coordinates in [um] 
  double m_coor_x[N2X1_IN_DET][ROWS2X1][COLS2X1];

  /// Array of y pixel coordinates in [um]
  double m_coor_y[N2X1_IN_DET][ROWS2X1][COLS2X1];

  /// Array of z pixel coordinates in [um]
 
  double m_coor_z[N2X1_IN_DET][ROWS2X1][COLS2X1];

  /// Minimal x coordinate of the pixel in [um]
  double m_coor_x_min;

  /// Maximal x coordinate of the pixel in [um]
  double m_coor_x_max;

  /// Minimal y coordinate of the pixel in [um] 
  double m_coor_y_min;

  /// Maximal y coordinate of the pixel in [um]
  double m_coor_y_max;

  /// Minimal z coordinate of the pixel in [um]
  double m_coor_z_min;

  /// Maximal z coordinate of the pixel in [um]
  double m_coor_z_max;

  /// Copy constructor
  PixCoordsCSPadV2 ( const PixCoordsCSPadV2& ) ;

  /// Assignment constructor
  PixCoordsCSPadV2& operator = ( const PixCoordsCSPadV2& ) ;
};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDSCSPADV2_H
