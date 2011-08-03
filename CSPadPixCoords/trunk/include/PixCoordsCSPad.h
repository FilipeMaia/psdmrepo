#ifndef CSPADPIXCOORDS_PIXCOORDSCSPAD_H
#define CSPADPIXCOORDS_PIXCOORDSCSPAD_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsCSPad.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "CSPadPixCoords/PixCoordsQuad.h"
#include "PSCalib/CSPadCalibPars.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace CSPadPixCoords {

/**
 *
 *
 *
 *
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


class PixCoordsCSPad  {
public:

  enum { NQuadsInCSPad = 4 };
  enum { N2x1InQuad    = 8 };
  enum { NCols2x1      = Psana::CsPad::ColumnsPerASIC     }; // 185
  enum { NRows2x1      = Psana::CsPad::MaxRowsPerASIC * 2 }; // 194*2 = 388

  // Default constructor
  PixCoordsCSPad ( PixCoordsQuad *pix_coords_quad,  PSCalib::CSPadCalibPars *cspad_calibpar ) ;

  // Destructor
  virtual ~PixCoordsCSPad () ;

  void fillAllQuadCoordsInCSPad() ;
  void fillOneQuadCoordsInCSPad(uint32_t quad) ;
  void setConstXYMinMax() ;
  void fillArrsOfCSPadPixCoords() ;

  float getPixCoor_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;
  float getPixCoor_pix(CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;
  float getPixCoor_int(CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;

  float*    getPixCoorArrX_um (){return &m_coor_x    [0][0][0][0];}
  float*    getPixCoorArrY_um (){return &m_coor_y    [0][0][0][0];}
  float*    getPixCoorArrX_pix(){return &m_coor_x_pix[0][0][0][0];}
  float*    getPixCoorArrY_pix(){return &m_coor_y_pix[0][0][0][0];}
  uint32_t* getPixCoorArrX_int(){return &m_coor_x_int[0][0][0][0];}
  uint32_t* getPixCoorArrY_int(){return &m_coor_y_int[0][0][0][0];}

private:

  // Data members
  PixCoordsQuad           *m_pix_coords_quad;  
  PSCalib::CSPadCalibPars *m_cspad_calibpar;  

  float    m_coor_x    [NQuadsInCSPad][N2x1InQuad][NCols2x1][NRows2x1];
  float    m_coor_y    [NQuadsInCSPad][N2x1InQuad][NCols2x1][NRows2x1];
  float    m_coor_x_pix[NQuadsInCSPad][N2x1InQuad][NCols2x1][NRows2x1];
  float    m_coor_y_pix[NQuadsInCSPad][N2x1InQuad][NCols2x1][NRows2x1];
  uint32_t m_coor_x_int[NQuadsInCSPad][N2x1InQuad][NCols2x1][NRows2x1];
  uint32_t m_coor_y_int[NQuadsInCSPad][N2x1InQuad][NCols2x1][NRows2x1];

  float    m_xmin_quad [NQuadsInCSPad]; 
  float    m_ymin_quad [NQuadsInCSPad]; 

  float m_coor_x_min;
  float m_coor_y_min;
  float m_coor_x_max;
  float m_coor_y_max;

  // Copy constructor and assignment are disabled by default
  PixCoordsCSPad ( const PixCoordsCSPad& ) ;
  PixCoordsCSPad& operator = ( const PixCoordsCSPad& ) ;

  enum { NX_CSPAD = 1750,   // Image sizes are used to get constant m_coor_y_max and m_coor_y_max 
         NY_CSPAD = 1750 }; // In this class we do not need in these image sizes...
};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDSCSPAD_H
