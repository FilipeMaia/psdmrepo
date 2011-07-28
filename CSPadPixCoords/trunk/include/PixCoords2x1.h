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

/**
 *  Defines the X,Y,Z pixel coordinates for a single 2x1 in its natural (ONLINE) frame.
 *  getPixCoorRot...(...) methods return these coordinates for rotated by N*90 degree 2x1
 *  in um(micrometer) or pixels.
 *  
 *  Rows and columns are defined like in ONLINE:
 *  /reg/g/psdm/sw/external/lusi-xtc/2.12.0a/x86_64-rhel5-gcc41-opt/pdsdata/cspad/ElementIterator.hh,
 *  Detector.hh
 *  
 *  
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
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

  PixCoords2x1 (float row_size_um     = 109.92,  // pixel size of the row in um
                float col_size_um     = 109.92,  // pixel size of the column in um
                float gap_row_size_um = 274.8) ; // pixel size of the gap column in um (two columns like this in the gap)

  // Destructor
  virtual ~PixCoords2x1 () ;

  // Methods
  void fill_pix_coords_2x1 () ;

  float getPixCoorRot000_um  (COORDINATE icoor, unsigned row, unsigned col) ;
  float getPixCoorRot090_um  (COORDINATE icoor, unsigned row, unsigned col) ;
  float getPixCoorRot180_um  (COORDINATE icoor, unsigned row, unsigned col) ;
  float getPixCoorRot270_um  (COORDINATE icoor, unsigned row, unsigned col) ;
 
  float getPixCoorRot000_pix (COORDINATE icoor, unsigned row, unsigned col) ;
  float getPixCoorRot090_pix (COORDINATE icoor, unsigned row, unsigned col) ;
  float getPixCoorRot180_pix (COORDINATE icoor, unsigned row, unsigned col) ;
  float getPixCoorRot270_pix (COORDINATE icoor, unsigned row, unsigned col) ;

  float getPixCoorRotN90_um  (ORIENTATION n90, COORDINATE icoor, unsigned row, unsigned col) ;
  float getPixCoorRotN90_pix (ORIENTATION n90, COORDINATE icoor, unsigned row, unsigned col) ;
  float getPixCoorRotN90     (UNITS units, ORIENTATION n90, COORDINATE icoor, unsigned row, unsigned col) ;

  void print_member_data () ;
  void print_selected_coords_2x1 (ARRAXIS arraxis) ;

protected:

private:

  // Data members

  float  m_row_size_um;
  float  m_col_size_um;
  float  m_gap_row_size_um;
  float  m_gap_size_um;
  float  m_coor_row_max;
  float  m_coor_col_max;

  // Cols and rows are interchanged in order to have an order of arrays like in ONLINE.
  float  m_coor_row[NRows2x1];  
  float  m_coor_col[NCols2x1];  
  float  m_coor_ort[NCols2x1][NRows2x1];  

  float  k_row_um_to_pix;
  float  k_col_um_to_pix;
  float  k_ort_um_to_pix;

  // Copy constructor and assignment are disabled by default
  PixCoords2x1 ( const PixCoords2x1& ) ;
  PixCoords2x1& operator = ( const PixCoords2x1& ) ;
};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDS2X1_H
