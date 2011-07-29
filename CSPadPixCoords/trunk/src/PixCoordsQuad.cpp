//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsQuad...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/PixCoordsQuad.h"

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
PixCoordsQuad::PixCoordsQuad ( PixCoords2x1 *pix_coords_2x1,  PSCalib::CSPadCalibPars *cspad_calibpar )
  : m_pix_coords_2x1(pix_coords_2x1)
  , m_cspad_calibpar(cspad_calibpar)
{
  cout << "PixCoordsQuad::PixCoordsQuad:" << endl;
  m_pix_coords_2x1 -> print_member_data(); 
  m_cspad_calibpar -> printCalibPars();
  
  fillAllQuads();
}

//--------------

void PixCoordsQuad::fillAllQuads()
{
        for (uint32_t quad=0; quad < NQuadsInCSPad; ++quad)
          {
	    cout << "\n quad=" << quad << ":\n" ;
            fillOneQuad(quad);
	  }
}

//--------------

void PixCoordsQuad::fillOneQuad(uint32_t quad)
{
        for (uint32_t sect=0; sect < N2x1InQuad; ++sect)
          {
            //bool bitIsOn = roiMask & (1<<sect);
            //if( !bitIsOn ) continue;

            float xcenter  = m_cspad_calibpar -> getQuadMargX  ()
                           + m_cspad_calibpar -> getCenterX    (quad,sect)
                           + m_cspad_calibpar -> getCenterCorrX(quad,sect);

            float ycenter  = m_cspad_calibpar -> getQuadMargY  ()
                           + m_cspad_calibpar -> getCenterY    (quad,sect)
                           + m_cspad_calibpar -> getCenterCorrY(quad,sect);

            float zcenter  = m_cspad_calibpar -> getQuadMargZ  ()
                           + m_cspad_calibpar -> getCenterZ    (quad,sect)
                           + m_cspad_calibpar -> getCenterCorrZ(quad,sect);

            float rotation = m_cspad_calibpar -> getRotation   (quad,sect);

            float tilt     = m_cspad_calibpar -> getTilt       (quad,sect);

            fillOneSectionInQuad(quad, sect, xcenter, ycenter, zcenter, rotation, tilt);

	    //cout << " sect=" << sect;
          }
	//cout << endl;
}

//--------------

void PixCoordsQuad::fillOneSectionInQuad(uint32_t quad, uint32_t sect, float xcenter, float ycenter, float zcenter, float rotation, float tilt)
{
  //cout << "PixCoordsQuad::fillOneSectionInQuad" << endl;
  cout << " sect=" << sect;

  //if (sect != 0) return;

            PixCoords2x1::ORIENTATION orient = PixCoords2x1::getOrientation(rotation);
            float pixSize_um = PSCalib::CSPadCalibPars::getRowSize_um();

            float ymin = xcenter*pixSize_um - m_pix_coords_2x1->getYCenterOffset_um(orient);
            float xmin = ycenter*pixSize_um - m_pix_coords_2x1->getXCenterOffset_um(orient);

            cout << " rotat="    << rotation;
            cout << " NRows2x1=" << NRows2x1;
            cout << " NCols2x1=" << NCols2x1;
            cout << " xcenter="  << xcenter;
            cout << " ycenter="  << ycenter;
            cout << " Xoffset="  << m_pix_coords_2x1->getXCenterOffset_um(orient);
            cout << " Yoffset="  << m_pix_coords_2x1->getYCenterOffset_um(orient);
            cout << " xmin="     << xmin;
            cout << " ymin="     << ymin;
            cout << endl;

            CSPadPixCoords::PixCoords2x1::COORDINATE X = CSPadPixCoords::PixCoords2x1::X;
            CSPadPixCoords::PixCoords2x1::COORDINATE Y = CSPadPixCoords::PixCoords2x1::Y;

            for (uint32_t col=0; col<NCols2x1; col++) {
            for (uint32_t row=0; row<NRows2x1; row++) {

               m_coor_x[quad][sect][col][row] = xmin + m_pix_coords_2x1->getPixCoorRotN90_um (orient, X, row, col);
               m_coor_y[quad][sect][col][row] = ymin + m_pix_coords_2x1->getPixCoorRotN90_um (orient, Y, row, col);

            }
            }
}

//--------------

float PixCoordsQuad::getPixCoorRot000_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return m_coor_x[quad][sect][col][row];
    case CSPadPixCoords::PixCoords2x1::Y : return m_coor_y[quad][sect][col][row];
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

float PixCoordsQuad::getPixCoorRot000_pix (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return m_coor_x[quad][sect][col][row] * PSCalib::CSPadCalibPars::getRowUmToPix();
    case CSPadPixCoords::PixCoords2x1::Y : return m_coor_y[quad][sect][col][row] * PSCalib::CSPadCalibPars::getColUmToPix();
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------
// Destructor --
//--------------
PixCoordsQuad::~PixCoordsQuad ()
{
}

} // namespace CSPadPixCoords
