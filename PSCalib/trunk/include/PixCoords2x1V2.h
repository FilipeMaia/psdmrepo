#ifndef PSCALIB_PIXCOORDS2X1V2_H
#define PSCALIB_PIXCOORDS2X1V2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoords2x1V2.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
//#include "psddl_psana/cspad.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSCalib {


/// @addtogroup PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief PixCoords2x1V2 class defines the 2x1 sensor pixel coordinates in its local frame. V2 stands for latest version.
 *  This module was copied from packege CSPadPixCoords in order to get rid of cyclic dependences
 *
 *
 *
 *  2x1 sensor coordinate frame:
 * 
 *  @code
 *    (Xmin,Ymax)      ^ Y          (Xmax,Ymax)
 *    (0,0)            |            (0,387)
 *       ------------------------------
 *       |             |              |
 *       |             |              |
 *       |             |              |
 *     --|-------------+--------------|----> X
 *       |             |              |
 *       |             |              |
 *       |             |              |
 *       ------------------------------
 *    (184,0)          |           (184,387)
 *    (Xmin,Ymin)                  (Xmax,Ymin)
 *  @endcode
 *
 *  The (r,c)=(0,0) is in the top left corner of the matrix which has coordinates (Xmin,Ymax)
 *  Here we assume that 2x1 has 185 rows and 388 columns.
 *  This assumption differs from the DAQ map, where rows and cols are interchanged:
 *  /reg/g/psdm/sw/external/lusi-xtc/2.12.0a/x86_64-rhel5-gcc41-opt/pdsdata/cspad/ElementIterator.hh,
 *  Detector.hh
 *   
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Include and typedef
 *  @code
 *  #include "PSCalib/PixCoords2x1V2.h"
 *  typedef PSCalib::PixCoords2x1V2 PC2X1;
 *  @endcode
 *
 *  @li  Instatiation
 *  @code
 *       PC2X1 *pix_coords_2x1 = new PC2X1();  
 *  or
 *       bool use_wide_pix_center = true;
 *       PC2X1 *pix_coords_2x1 = new PC2X1(use_wide_pix_center);  
 *  @endcode
 *
 *  @li  Printing methods
 *  @code
 *       pix_coords_2x1 -> print_member_data();
 *       pix_coords_2x1 -> print_coord_arrs_2x1();
 *
 *       double angle = 5;
 *       pix_coords_2x1 -> print_map_min_max(PC2X1::PIX);
 *       pix_coords_2x1 -> print_map_min_max(PC2X1::PIX, angle);
 *       pix_coords_2x1 -> print_map_min_max(PC2X1::UM);
 *       pix_coords_2x1 -> print_map_min_max(PC2X1::UM, angle);
 *  @endcode
 *
 *  @li  Access methods
 *  @code
 *       double* x_arr = pix_coords_2x1 -> get_x_map_2x1_pix ();
 *       double* y_arr = pix_coords_2x1 -> get_y_map_2x1_pix ();
 *       double* z_arr = pix_coords_2x1 -> get_z_map_2x1_pix (); // returns 0-s
 *  or
 *       double* x_arr = pix_coords_2x1 -> get_coord_map_2x1 (PC2X1::AXIS_X, PC2X1::UM, angle);
 *       double* y_arr = pix_coords_2x1 -> get_coord_map_2x1 (PC2X1::AXIS_Y, PC2X1::UM, angle);
 *       double* z_arr = pix_coords_2x1 -> get_coord_map_2x1 (PC2X1::AXIS_Z, PC2X1::UM, angle); // returns 0-s
 *
 *       double x_min_um  = pix_coords_2x1 -> get_min_of_coord_map_2x1 (PC2X1::AXIS_X, PC2X1::UM,  angle);
 *       double y_min_um  = pix_coords_2x1 -> get_min_of_coord_map_2x1 (PC2X1::AXIS_Y, PC2X1::UM,  angle);
 *       double z_min_um  = pix_coords_2x1 -> get_min_of_coord_map_2x1 (PC2X1::AXIS_Z, PC2X1::UM,  angle);
 *       double y_max_pix = pix_coords_2x1 -> get_max_of_coord_map_2x1 (PC2X1::AXIS_Y, PC2X1::PIX, angle);
 *  @endcode
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *
 *  @version \$Id$
 *
 *  @author Mikhail S. Dubrovin
 */ 

// *  @see CSPadImageProducer, PixCoordsTest


class PixCoords2x1V2  {
public:

  /// Enumerator for X, Y, and Z axes
  enum AXIS { AXIS_X=0,
              AXIS_Y,
              AXIS_Z };

  /// Enumerator for scales, coordinates in micrometers or pixels
  enum UNITS { UM=0,
               PIX };
 
  /// Number of pixel rows in 2x1 
  const static unsigned ROWS2X1     = 185;

  /// Number of pixel columnss in 2x1
  const static unsigned COLS2X1     = 388;

  /// Half number of pixel columnss in 2x1
  const static unsigned COLS2X1HALF = 194;

  /// Number of pixels in 2x1
  const static unsigned SIZE2X1     = COLS2X1*ROWS2X1; 

  /// Number of corners...
  const static unsigned NCORNERS    =   4;

  /// Pixel size [um] in column direction
  const static double PIX_SIZE_COLS = 109.92;

  /// Pixel size [um] in row direction
  const static double PIX_SIZE_ROWS = 109.92;

  /// Wide pixel length [um] 
  const static double PIX_SIZE_WIDE = 274.80;

  /// Pixel size [um]  
  const static double PIX_SIZE_UM   = 109.92;

  /// Conversion factor between um and pix 
  const static double UM_TO_PIX     = 1./109.92;

  // Constructor

  /**
   *  @brief Fills-in the map of perfect 2x1 coordinates, defined through the chip geometry.
   *  @param[in] use_wide_pix_center Optional parameter can be used if the wide-pixel row coordinate is prefered to be in the raw center.
   */
  PixCoords2x1V2 (bool use_wide_pix_center=false);

  /// Destructor
  virtual ~PixCoords2x1V2 ();

  /// Prints 2x1 pixel coordinates
  void print_coord_arrs_2x1();


  /// Prints class member data
  void print_member_data ();

 /**  
   *  @brief Prints minimal and maximal values of the 2x1 coordinate map
   *  @param[in] units      Units [UM] or [PIX] from the enumerated list
   *  @param[in] angle_deg  2x1 rotation angle [degree] 
   */
  void print_map_min_max(UNITS units, const double& angle_deg);

  // Access methods

  /// Returns pointer to the 2x1 pixel map of x-coordinate [um]
  double* get_x_map_2x1_um  () { return &m_x_map_2x1_um[0][0]; } 

  /// Returns pointer to the 2x1 pixel map of y-coordinate [um]
  double* get_y_map_2x1_um  () { return &m_y_map_2x1_um[0][0]; } 

  /// Returns pointer to the 2x1 pixel map of z-coordinate [um]
  double* get_z_map_2x1_um  () { return &m_z_map_2x1[0][0]; } 

  /// Returns pointer to the 2x1 pixel map of x-coordinate [pix]
  double* get_x_map_2x1_pix () { return &m_x_map_2x1_pix[0][0]; } 

  /// Returns pointer to the 2x1 pixel map of y-coordinate [pix]
  double* get_y_map_2x1_pix () { return &m_y_map_2x1_pix[0][0]; } 

  /// Returns pointer to the 2x1 pixel map of z-coordinate [pix]
  double* get_z_map_2x1_pix () { return &m_z_map_2x1[0][0]; } 

  /// Returns sizee of the coordinate arrays
  const unsigned get_size() {return unsigned(SIZE2X1);}

  /**  
   *  @brief Returns pointer to the 2x1 pixel map for specified parameters
   *  @param[in] axis       Axis from the enumerated list for X, Y, and Z
   *  @param[in] units      Units [UM] or [PIX] from the enumerated list
   *  @param[in] angle_deg  2x1 rotation angle [degree] 
   */
  double* get_coord_map_2x1        (AXIS axis, UNITS units, const double& angle_deg=0); 

  /// Returns minimal value of the 2x1 pixel coordinate for specified parameters
  double  get_min_of_coord_map_2x1 (AXIS axis, UNITS units, const double& angle_deg=0);  

  /// Returns miximal value of the 2x1 pixel coordinate for specified parameters
  double  get_max_of_coord_map_2x1 (AXIS axis, UNITS units, const double& angle_deg=0);  


protected:

  /// Generator of the 2x1 pixel coordinate map.
  void make_maps_of_2x1_pix_coordinates ();


private:

  bool m_use_wide_pix_center; /// switch between two options of the wide pixel row center
  double  m_angle_deg;        /// 2x1 rotation angle [degree] 
  UNITS   m_units;            /// Units [UM] or [PIX] from the enumerated list

  // Cols and rows are interchanged in order to have an order of arrays like in ONLINE.
  double  m_x_rhs[COLS2X1HALF];  

  double  m_x_arr_um [COLS2X1];  
  double  m_y_arr_um [ROWS2X1];  
  double  m_x_arr_pix[COLS2X1];  
  double  m_y_arr_pix[ROWS2X1];  

  const static unsigned IND_CORNER[NCORNERS];

  double  m_z_map_2x1    [ROWS2X1][COLS2X1]; // contains 0-s and works for all - _um, _pix, and _rot
  double  m_x_map_2x1_um [ROWS2X1][COLS2X1];  
  double  m_y_map_2x1_um [ROWS2X1][COLS2X1];  
  double  m_x_map_2x1_pix[ROWS2X1][COLS2X1];  
  double  m_y_map_2x1_pix[ROWS2X1][COLS2X1];  
  double  m_x_map_2x1_rot[ROWS2X1][COLS2X1];  
  double  m_y_map_2x1_rot[ROWS2X1][COLS2X1];  

  // Copy constructor and assignment are disabled by default
  PixCoords2x1V2 ( const PixCoords2x1V2& ) ;
  PixCoords2x1V2& operator = ( const PixCoords2x1V2& ) ;
};

const static double DEG_TO_RAD = 3.141592653589793238463 / 180; 

/// Global method for x and y arrays rotation
void rotation(const double* x, const double* y, unsigned size, double angle_deg,   double* xrot, double* yrot);
/// Global method for x and y arrays rotation
void rotation(const double* x, const double* y, unsigned size, double C, double S, double* xrot, double* yrot);
/// Global method, returns minimal value of the array of specified length 
double min_of_array(const double* arr, unsigned size);
/// Global method, returns maximal value of the array of specified length 
double max_of_array(const double* arr, unsigned size);

} // namespace PSCalib

#endif // PSCALIB_PIXCOORDS2X1V2_H
