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
PixCoordsCSPad::PixCoordsCSPad ( PixCoordsQuad *pix_coords_quad,  PSCalib::CSPadCalibPars *cspad_calibpar )
  : m_pix_coords_quad(pix_coords_quad)
  , m_cspad_calibpar (cspad_calibpar)
{
  cout << "PixCoordsQuad::PixCoordsCSPad:" << endl;
  //m_pix_coords_quad -> print_member_data(); 
  //m_cspad_calibpar  -> printCalibPars();

  fillAllQuadCoordsInCSPad();

}

//--------------

void PixCoordsCSPad::fillAllQuadCoordsInCSPad()
{
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
	    cout << "\n quad=" << q << ":\n" ;

            m_xmin_quad [q] = m_cspad_calibpar -> getOffsetX    (q) 
                            + m_cspad_calibpar -> getOffsetCorrX(q)  
                            + dx[q];

            m_ymin_quad [q] = m_cspad_calibpar -> getOffsetY    (q) 
                            + m_cspad_calibpar -> getOffsetCorrY(q) 
                            + dy[q];            
 
            m_xmin_quad [q] *= pixSize_um;
            m_ymin_quad [q] *= pixSize_um;
	    

	    cout << "  m_xmin_quad = " << m_xmin_quad [q]
	         << "  m_ymin_quad = " << m_ymin_quad [q];


            fillOneQuadCoordsInCSPad(q);
	  }
}

//--------------

void PixCoordsCSPad::fillOneQuadCoordsInCSPad(uint32_t quad)
{
        CSPadPixCoords::PixCoords2x1::COORDINATE X = CSPadPixCoords::PixCoords2x1::X;
        CSPadPixCoords::PixCoords2x1::COORDINATE Y = CSPadPixCoords::PixCoords2x1::Y;

        float rotation = m_cspad_calibpar -> getQuadRotation(quad);
	cout << "  Quad rotation = " <<  rotation << endl;

        CSPadPixCoords::PixCoords2x1::ORIENTATION orient = PixCoords2x1::getOrientation(rotation);

        for (uint32_t sect=0; sect<N2x1InQuad; ++sect) {
            for (uint32_t row=0; row<NRows2x1; row++) {
            for (uint32_t col=0; col<NCols2x1; col++) {

	       float coor_x = m_xmin_quad[quad] + m_pix_coords_quad->getPixCoorRotN90_um (orient, X, quad, sect, row, col);
	       float coor_y = m_ymin_quad[quad] + m_pix_coords_quad->getPixCoorRotN90_um (orient, Y, quad, sect, row, col);
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

float PixCoordsCSPad::getPixCoor_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return m_coor_x[quad][sect][col][row];// - m_coor_x_min;
    case CSPadPixCoords::PixCoords2x1::Y : return m_coor_y[quad][sect][col][row];// - m_coor_y_min;
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

float PixCoordsCSPad::getPixCoor_pix(CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return getPixCoor_um (icoor, quad, sect, row, col) * PSCalib::CSPadCalibPars::getRowUmToPix();
    case CSPadPixCoords::PixCoords2x1::Y : return getPixCoor_um (icoor, quad, sect, row, col) * PSCalib::CSPadCalibPars::getColUmToPix();
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
