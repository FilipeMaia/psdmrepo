#ifndef CSPADPIXCOORDS_PIXCOORDSQUAD_H
#define CSPADPIXCOORDS_PIXCOORDSQUAD_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsQuad.
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

/**
 *
 *
 *
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

class PixCoordsQuad  {
public:

  enum { NQuadsInCSPad = 4 };
  enum { N2x1InQuad    = Psana::CsPad::SectorsPerQuad     }; // 8
  enum { NCols2x1      = Psana::CsPad::ColumnsPerASIC     }; // 185
  enum { NRows2x1      = Psana::CsPad::MaxRowsPerASIC * 2 }; // 194*2 = 388

  // Default constructor
  PixCoordsQuad ( PixCoords2x1 *pix_coords_2x1,  PSCalib::CSPadCalibPars *cspad_calibpar ) ;

  // Destructor
  virtual ~PixCoordsQuad () ;


  void fillAllQuads();
  void fillOneQuad(uint32_t quad);
  void fillOneSectionInQuad      (uint32_t quad, uint32_t sect, float xcenter, float ycenter, float zcenter, float rotation);
  void fillOneSectionTiltedInQuad(uint32_t quad, uint32_t sect, float xcenter, float ycenter, float zcenter, float rotation, float tilt);
  void setConstXYMinMax();

  float getPixCoorRot000_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;
  float getPixCoorRot090_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;
  float getPixCoorRot180_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;
  float getPixCoorRot270_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;

  float getPixCoorRot000_pix(CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;
  float getPixCoorRot090_pix(CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;
  float getPixCoorRot180_pix(CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;
  float getPixCoorRot270_pix(CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;

  float getPixCoorRotN90_um (CSPadPixCoords::PixCoords2x1::ORIENTATION orient, 
                             CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;
  float getPixCoorRotN90_pix(CSPadPixCoords::PixCoords2x1::ORIENTATION orient, 
                             CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;
  float getPixCoorRotN90    (CSPadPixCoords::PixCoords2x1::UNITS units,
                             CSPadPixCoords::PixCoords2x1::ORIENTATION orient, 
                             CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col) ;

protected:

private:

  // Data members
  CSPadPixCoords::PixCoords2x1::COORDINATE XCOOR;
  CSPadPixCoords::PixCoords2x1::COORDINATE YCOOR;
  CSPadPixCoords::PixCoords2x1::COORDINATE ZCOOR;

  float m_degToRad; 

  PixCoords2x1            *m_pix_coords_2x1;  
  PSCalib::CSPadCalibPars *m_cspad_calibpar;  

  float m_coor_x[NQuadsInCSPad][N2x1InQuad][NCols2x1][NRows2x1];
  float m_coor_y[NQuadsInCSPad][N2x1InQuad][NCols2x1][NRows2x1];

  float m_coor_x_min[NQuadsInCSPad];
  float m_coor_x_max[NQuadsInCSPad];
  float m_coor_y_min[NQuadsInCSPad];
  float m_coor_y_max[NQuadsInCSPad];

  // Copy constructor and assignment are disabled by default
  PixCoordsQuad ( const PixCoordsQuad& ) ;
  PixCoordsQuad& operator = ( const PixCoordsQuad& ) ;

};

} // namespace CSPadPixCoords

#endif // CSPADPIXCOORDS_PIXCOORDSQUAD_H
