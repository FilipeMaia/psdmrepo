#ifndef CSPADPIXCOORDS_PIXCOORDS2X1V2_H
#define CSPADPIXCOORDS_PIXCOORDS2X1V2_H

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
#include "psddl_psana/cspad.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace CSPadPixCoords {

/// @addtogroup CSPadPixCoords

/**
 *  @ingroup CSPadPixCoords
 *
 *  @brief PixCoords2x1V2 class defines the 2x1 section pixel coordinates in its local frame.
 *
 *  Defines the X,Y pixel coordinates for single 2x1 in the frame:
 *  Assume that 2x1 has 195 rows and 388 columns
 *  The (r,c)=(0,0) is in the top left corner of the matrix, has coordinates (xmin,ymax)
 * 
 *                     ^ Y          (Xmax,Ymax)
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
 *    (Xmin,Ymin)
 *
 *
 *  DIFFERENT from the DAQ map... rows<->cols:
 *  /reg/g/psdm/sw/external/lusi-xtc/2.12.0a/x86_64-rhel5-gcc41-opt/pdsdata/cspad/ElementIterator.hh,
 *  Detector.hh
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CSPadImageProducer, PixCoordsTest
 *
 *  @version $Id$
 *
 * @author Mikhail S. Dubrovin
 *
 *
 */ 

const static double DEG_TO_RAD    = 3.14159265359 / 180; 

void rotation(const double* x, const double* y, const unsigned& size, const double& angle_deg,          double* xrot, double* yrot);
void rotation(const double* x, const double* y, const unsigned& size, const double& C, const double& S, double* xrot, double* yrot);
double min_of_array(const double* arr, const unsigned& size);
double max_of_array(const double* arr, const unsigned& size);


class PixCoords2x1V2  {
public:

  enum AXIS { X=0, 
              Y,
              Z };

  enum UNITS { UM=0,
               PIX };  // coordinates in micrometers or pixels

  const static int    ROWS2X1       = 185;    // Psana::CsPad::ColumnsPerASIC     };
  const static int    COLS2X1       = 388;    // Psana::CsPad::MaxRowsPerASIC * 2 };
  const static int    COLS2X1HALF   = 194;    // Psana::CsPad::MaxRowsPerASIC     };
  const static int    SIZE2X1       = COLS2X1*ROWS2X1;
  const static int    NCORNERS      =   4;

  const static double PIX_SIZE_COLS = 109.92;
  const static double PIX_SIZE_ROWS = 109.92;
  const static double PIX_SIZE_WIDE = 274.80;

  // Default constructor
  /**
   *  @brief No parameters needed; everything is defined through the fixed 2x1 chip geometry.
   *  
   *  Fills/holds/provides access to the arrays of row, column, and ortogonal coordinate of 2x1 pixels
   */
  PixCoords2x1V2 (bool use_wide_pix_center=false);

  // Destructor
  virtual ~PixCoords2x1V2 ();

  // Methods
  void print_coord_arrs_2x1();
  void print_member_data ();
  void print_map_min_max(UNITS units, const double& angle_deg);

  /**
   *  Access methods return the coordinate for indicated axis, 2x1 row and column
   *  indexes after the 2x1 rotation by n*90 degree.
   *  The pixel coordinates can be returned in um(micrometer) and pix(pixel).
   */

  double* get_x_map_2x1_um  () { return &m_x_map_2x1_um [0][0]; } 
  double* get_y_map_2x1_um  () { return &m_y_map_2x1_um [0][0]; } 
  double* get_x_map_2x1_pix () { return &m_x_map_2x1_pix[0][0]; } 
  double* get_y_map_2x1_pix () { return &m_y_map_2x1_pix[0][0]; } 

  double* get_coord_map_2x1        (AXIS axis, UNITS units, const double& angle_deg); 
  double  get_min_of_coord_map_2x1 (AXIS axis, UNITS units, const double& angle_deg);  
  double  get_max_of_coord_map_2x1 (AXIS axis, UNITS units, const double& angle_deg);  

protected:

  void make_maps_of_2x1_pix_coordinates ();


private:

  bool m_use_wide_pix_center;
  double  m_angle_deg;  
  UNITS   m_units;  

  // Cols and rows are interchanged in order to have an order of arrays like in ONLINE.
  double  m_x_rhs[COLS2X1HALF];  

  double  m_x_arr_um [COLS2X1];  
  double  m_y_arr_um [ROWS2X1];  
  double  m_x_arr_pix[COLS2X1];  
  double  m_y_arr_pix[ROWS2X1];  

  const static unsigned IND_CORNER[NCORNERS];

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

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDS2X1V2_H
