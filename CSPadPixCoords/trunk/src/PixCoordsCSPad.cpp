//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsCSPad...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/PixCoordsCSPad.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <math.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------
PixCoordsCSPad::PixCoordsCSPad ( PixCoordsQuad *pix_coords_quad,  PSCalib::CSPadCalibPars *cspad_calibpar, bool tiltIsApplied )
  : m_pix_coords_quad(pix_coords_quad)
  , m_cspad_calibpar (cspad_calibpar)
  , m_tiltIsApplied  (tiltIsApplied)
{
  cout << "PixCoordsQuad::PixCoordsCSPad" << endl;
  //m_pix_coords_quad -> print_member_data(); 
  //m_cspad_calibpar  -> printCalibPars();

  XCOOR = CSPadPixCoords::PixCoords2x1::X;
  YCOOR = CSPadPixCoords::PixCoords2x1::Y;
  ZCOOR = CSPadPixCoords::PixCoords2x1::Z;

  fillAllQuadCoordsInCSPad();
  //setConstXYMinMax();
  fillArrsOfCSPadPixCoords();
}

//--------------

void PixCoordsCSPad::fillAllQuadCoordsInCSPad()
{
        m_degToRad = 3.14159265359 / 180.; 

        m_coor_x_min = 1e5;
        m_coor_x_max = 0;
        m_coor_y_min = 1e5;
        m_coor_y_max = 0;

        float pixSize_um = PSCalib::CSPadCalibPars::getRowSize_um();

        float margX  = m_cspad_calibpar -> getMargX  ();
        float margY  = m_cspad_calibpar -> getMargY  ();
        float gapX   = m_cspad_calibpar -> getGapX   ();
        float gapY   = m_cspad_calibpar -> getGapY   ();
        float shiftX = m_cspad_calibpar -> getShiftX ();
        float shiftY = m_cspad_calibpar -> getShiftY ();

        float dx[] = {margX-gapX+shiftX,  margX-gapX-shiftX,  margX+gapX-shiftX,  margX+gapX+shiftX};
        float dy[] = {margY-gapY-shiftY,  margY+gapY-shiftY,  margY+gapY+shiftY,  margY-gapY+shiftY};

        for (uint32_t q=0; q < NQuadsInCSPad; ++q)
          {
	    //cout << "quad=" << q << ":" ;

            m_xmin_quad [q] = m_cspad_calibpar -> getOffsetX    (q) 
                            + m_cspad_calibpar -> getOffsetCorrX(q)  
                            + dx[q];

            m_ymin_quad [q] = m_cspad_calibpar -> getOffsetY    (q) 
                            + m_cspad_calibpar -> getOffsetCorrY(q) 
                            + dy[q];            
 
            m_xmin_quad [q] *= pixSize_um;
            m_ymin_quad [q] *= pixSize_um;
	    
	    //cout << "  m_xmin_quad = " << m_xmin_quad [q]
	    //     << "  m_ymin_quad = " << m_ymin_quad [q];


            if (m_tiltIsApplied) fillOneQuadTiltedCoordsInCSPad(q);
            else                 fillOneQuadCoordsInCSPad(q);
	  }
}

//--------------

void PixCoordsCSPad::fillOneQuadCoordsInCSPad(uint32_t quad)
{
        float rotation = m_cspad_calibpar -> getQuadRotation(quad);
	// cout << "  Quad rotation = " <<  rotation << endl;

        CSPadPixCoords::PixCoords2x1::ORIENTATION orient = PixCoords2x1::getOrientation(rotation);

        for (uint32_t sect=0; sect<N2x1InQuad; ++sect) {
            for (uint32_t row=0; row<NRows2x1; row++) {
            for (uint32_t col=0; col<NCols2x1; col++) {

	       float coor_x = m_xmin_quad[quad] + m_pix_coords_quad->getPixCoorRotN90_um (orient, XCOOR, quad, sect, row, col);
	       float coor_y = m_ymin_quad[quad] + m_pix_coords_quad->getPixCoorRotN90_um (orient, YCOOR, quad, sect, row, col);

	       m_coor_x[quad][sect][col][row] = coor_x;
	       m_coor_y[quad][sect][col][row] = coor_y;

               if ( coor_x < m_coor_x_min ) m_coor_x_min = coor_x;
               if ( coor_x > m_coor_x_max ) m_coor_x_max = coor_x;
               if ( coor_y < m_coor_y_min ) m_coor_y_min = coor_y;
               if ( coor_y > m_coor_y_max ) m_coor_y_max = coor_y;

            } // col
            } // row
        }   // sect
	//cout << endl;
}

//--------------

void PixCoordsCSPad::fillOneQuadTiltedCoordsInCSPad(uint32_t quad)
{
        float rotation = m_cspad_calibpar -> getQuadRotation(quad);
        float tilt     = m_cspad_calibpar -> getQuadTilt(quad);
        CSPadPixCoords::PixCoords2x1::ORIENTATION orient = PixCoords2x1::getOrientation(rotation);

	cout << "Quad:"       << quad
	     << "  rotation:" << rotation
	     << "  tilt:"     << tilt
	     << endl;

        float tiltrad = tilt * m_degToRad;
        float sintilt = sin(tiltrad);
        float costilt = cos(tiltrad);

        for (uint32_t sect=0; sect<N2x1InQuad; ++sect) {
            for (uint32_t row=0; row<NRows2x1; row++) {
            for (uint32_t col=0; col<NCols2x1; col++) {

	       float x_in_quad = m_pix_coords_quad->getPixCoorRotN90_um (orient, XCOOR, quad, sect, row, col);
	       float y_in_quad = m_pix_coords_quad->getPixCoorRotN90_um (orient, YCOOR, quad, sect, row, col);

               float x_tilted =  x_in_quad * costilt - y_in_quad * sintilt;
               float y_tilted =  x_in_quad * sintilt + y_in_quad * costilt; 

	       float coor_x = m_xmin_quad[quad] + x_tilted;
	       float coor_y = m_ymin_quad[quad] + y_tilted;

	       m_coor_x[quad][sect][col][row] = coor_x;
	       m_coor_y[quad][sect][col][row] = coor_y;

               if ( coor_x < m_coor_x_min ) m_coor_x_min = coor_x;
               if ( coor_x > m_coor_x_max ) m_coor_x_max = coor_x;
               if ( coor_y < m_coor_y_min ) m_coor_y_min = coor_y;
               if ( coor_y > m_coor_y_max ) m_coor_y_max = coor_y;

            } // col
            } // row
        }   // sect
	//cout << endl;
}

//--------------

void PixCoordsCSPad::setConstXYMinMax()
{
	      m_coor_x_min = 0;
	      m_coor_y_min = 0;
	      m_coor_x_max = NX_CSPAD * PSCalib::CSPadCalibPars::getRowSize_um();
	      m_coor_y_max = NY_CSPAD * PSCalib::CSPadCalibPars::getColSize_um();
}

//--------------

void PixCoordsCSPad::fillArrsOfCSPadPixCoords()
{
  for (uint32_t quad=0; quad<NQuadsInCSPad; quad++) {
    for (uint32_t sect=0; sect<N2x1InQuad;  sect++) {
      for (uint32_t row=0; row<NRows2x1;     row++) {
        for (uint32_t col=0; col<NCols2x1;   col++) {

	  float coor_x_um = m_coor_x[quad][sect][col][row] - m_coor_x_min;
	  float coor_y_um = m_coor_y[quad][sect][col][row] - m_coor_y_min;

          m_coor_x     [quad][sect][col][row] = coor_x_um;
          m_coor_y     [quad][sect][col][row] = coor_y_um;

          m_coor_x_pix [quad][sect][col][row] = coor_x_um * PSCalib::CSPadCalibPars::getRowUmToPix();
          m_coor_y_pix [quad][sect][col][row] = coor_y_um * PSCalib::CSPadCalibPars::getColUmToPix();

          m_coor_x_int [quad][sect][col][row] = (int) m_coor_x_pix [quad][sect][col][row];
          m_coor_y_int [quad][sect][col][row] = (int) m_coor_y_pix [quad][sect][col][row];
	}
      }
    }
  }
}

//--------------

float PixCoordsCSPad::getPixCoor_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return m_coor_x [quad][sect][col][row];
    case CSPadPixCoords::PixCoords2x1::Y : return m_coor_y [quad][sect][col][row];
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

float PixCoordsCSPad::getPixCoor_pix(CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return m_coor_x_pix [quad][sect][col][row];
    case CSPadPixCoords::PixCoords2x1::Y : return m_coor_y_pix [quad][sect][col][row]; 
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

float PixCoordsCSPad::getPixCoor_int(CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return m_coor_x_int [quad][sect][col][row];
    case CSPadPixCoords::PixCoords2x1::Y : return m_coor_y_int [quad][sect][col][row]; 
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

//--------------
// Destructor --
//--------------
PixCoordsCSPad::~PixCoordsCSPad ()
{
}

} // namespace CSPadPixCoords
