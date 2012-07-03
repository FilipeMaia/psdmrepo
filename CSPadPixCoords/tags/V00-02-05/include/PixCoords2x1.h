#ifndef CSPADPIXCOORDS_PIXCOORDS2X1_H
#define CSPADPIXCOORDS_PIXCOORDS2X1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoords2x1.
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
 *  @brief PixCoords2x1 class defines the 2x1 section pixel coordinates in its local frame.
 *
 *  Defines the X,Y,Z pixel coordinates for single 2x1 in its own (ONLINE) frame:
 *  X coordinate is directed along rows from 0 to 388 (from top to botton for 0 rotation angle).
 *  Y coordinate is directed opposite columns from 184 to 0 (from left to right for 0 rotation angle).
 *  getPixCoorRot...(...) methods return these coordinates for rotated by N*90 degree 2x1
 *  in um(micrometer) or pixels.
 *  
 *  Rows and columns are defined like in ONLINE:
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
 *  @author Mikhail S. Dubrovin
 */

class PixCoords2x1  {
public:
  enum ARRAXIS     { ROW = 0, 
                     COL, 
                     ORT };

  enum COORDINATE  { X = 0, 
                     Y, 
                     Z };

  enum ORIENTATION { R000 = 0, 
                     R090, 
                     R180, 
                     R270 };

  enum UNITS       { UM = 0,
                     PIX };  // coordinates in micrometers or pixels

  enum { NCols2x1  = Psana::CsPad::ColumnsPerASIC     }; // 185
  enum { NRows2x1  = Psana::CsPad::MaxRowsPerASIC * 2 }; // 194*2 = 388
  enum { NRowsASIC = Psana::CsPad::MaxRowsPerASIC     }; // 194

  // Default constructor
  /**
   *  @brief No parameters needed; everything is defined through the fixed 2x1 chip geometry.
   *  
   *  Fills/holds/provides access to the arrays of row, column, and ortogonal coordinate of 2x1 pixels
   */
  PixCoords2x1 () ;

  // Destructor
  virtual ~PixCoords2x1 () ;

  // Methods
  void fill_pix_coords_2x1 () ;

  /**
   *  Access methods return the coordinate for indicated axis, 2x1 row and column
   *  indexes after the 2x1 rotation by n*90 degree.
   *  The pixel coordinates can be returned in um(micrometer) and pix(pixel).
   */
  double getPixCoorRot000_um  (COORDINATE icoor, unsigned row, unsigned col) ;
  double getPixCoorRot090_um  (COORDINATE icoor, unsigned row, unsigned col) ;
  double getPixCoorRot180_um  (COORDINATE icoor, unsigned row, unsigned col) ;
  double getPixCoorRot270_um  (COORDINATE icoor, unsigned row, unsigned col) ;
 
  double getPixCoorRot000_pix (COORDINATE icoor, unsigned row, unsigned col) ;
  double getPixCoorRot090_pix (COORDINATE icoor, unsigned row, unsigned col) ;
  double getPixCoorRot180_pix (COORDINATE icoor, unsigned row, unsigned col) ;
  double getPixCoorRot270_pix (COORDINATE icoor, unsigned row, unsigned col) ;

  double getPixCoorRotN90_um  (ORIENTATION n90, COORDINATE icoor, unsigned row, unsigned col) ;
  double getPixCoorRotN90_pix (ORIENTATION n90, COORDINATE icoor, unsigned row, unsigned col) ;
  double getPixCoorRotN90     (UNITS units, ORIENTATION n90, COORDINATE icoor, unsigned row, unsigned col) ;

  static ORIENTATION getOrientation(double angle) ;
  static size_t getNCols     (ORIENTATION n90) ;
  static size_t getNRows     (ORIENTATION n90) ;
  double  getXCenterOffset_um (ORIENTATION n90) ; 
  double  getYCenterOffset_um (ORIENTATION n90) ; 

  void print_member_data () ;
  void print_selected_coords_2x1 (ARRAXIS arraxis) ;

protected:

private:

  // Data members
  double  m_row_size_um;
  double  m_col_size_um;
  double  m_gap_size_um;
  double  m_coor_row_max;
  double  m_coor_col_max;

  // Cols and rows are interchanged in order to have an order of arrays like in ONLINE.
  double  m_coor_row[NRows2x1];  
  double  m_coor_col[NCols2x1];  
  double  m_coor_ort[NCols2x1][NRows2x1];  

  double  k_center_of_rows_um; 
  double  k_center_of_cols_um; 
  double  k_center_of_rows_pix; 
  double  k_center_of_cols_pix; 

  double  k_row_um_to_pix;
  double  k_col_um_to_pix;
  double  k_ort_um_to_pix;

  // Copy constructor and assignment are disabled by default
  PixCoords2x1 ( const PixCoords2x1& ) ;
  PixCoords2x1& operator = ( const PixCoords2x1& ) ;
};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDS2X1_H
