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
#include <math.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include <boost/math/constants/constants.hpp>


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
PixCoordsQuad::PixCoordsQuad ( PixCoords2x1 *pix_coords_2x1,  PSCalib::CSPadCalibPars *cspad_calibpar, bool tiltIsApplied )
  : m_pix_coords_2x1(pix_coords_2x1)
  , m_cspad_calibpar(cspad_calibpar)
  , m_tiltIsApplied (tiltIsApplied)
{
  cout << "PixCoordsQuad::PixCoordsQuad" << endl;
  //m_pix_coords_2x1 -> print_member_data(); 
  //m_cspad_calibpar -> printCalibPars();

  XCOOR = CSPadPixCoords::PixCoords2x1::X;
  YCOOR = CSPadPixCoords::PixCoords2x1::Y;
  ZCOOR = CSPadPixCoords::PixCoords2x1::Z;

  fillAllQuads();
  setConstXYMinMax();
}

//--------------

void PixCoordsQuad::fillAllQuads()
{
        for (uint32_t quad=0; quad < NQuadsInCSPad; ++quad)
          {
	    //cout << "\n quad=" << quad << ":\n" ;

            m_coor_x_min[quad] = 1e5;
            m_coor_x_max[quad] = 0;
            m_coor_y_min[quad] = 1e5;
            m_coor_y_max[quad] = 0;

            m_degToRad = 3.14159265359 / 180.; 

            fillOneQuad(quad);
	  }
}

//--------------

void PixCoordsQuad::fillOneQuad(uint32_t quad)
{
        float pixSize_um = PSCalib::CSPadCalibPars::getRowSize_um();

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


            xcenter *= pixSize_um;
            ycenter *= pixSize_um;

            if (m_tiltIsApplied) fillOneSectionTiltedInQuad(quad, sect, xcenter, ycenter, zcenter, rotation, tilt);
            else                 fillOneSectionInQuad(quad, sect, xcenter, ycenter, zcenter, rotation);

	    //cout << " sect=" << sect;
          }
	//cout << endl;

}

//--------------

void PixCoordsQuad::fillOneSectionInQuad(uint32_t quad, uint32_t sect, float xcenter, float ycenter, float zcenter, float rotation)
{
  // cout << "PixCoordsQuad::fillOneSectionInQuad" << endl;
  // cout << " sect=" << sect;
  // if (sect != 0) return;

            PixCoords2x1::ORIENTATION orient = PixCoords2x1::getOrientation(rotation);

            float xmin = xcenter - m_pix_coords_2x1->getXCenterOffset_um(orient);
            float ymin = ycenter - m_pix_coords_2x1->getYCenterOffset_um(orient);

            for (uint32_t col=0; col<NCols2x1; col++) {
            for (uint32_t row=0; row<NRows2x1; row++) {

	       float coor_x = xmin + m_pix_coords_2x1->getPixCoorRotN90_um (orient, XCOOR, row, col);
	       float coor_y = ymin + m_pix_coords_2x1->getPixCoorRotN90_um (orient, YCOOR, row, col);
	       m_coor_x[quad][sect][col][row] = coor_x;
	       m_coor_y[quad][sect][col][row] = coor_y;

               if ( coor_x < m_coor_x_min[quad] ) m_coor_x_min[quad] = coor_x;
               if ( coor_x > m_coor_x_max[quad] ) m_coor_x_max[quad] = coor_x;
               if ( coor_y < m_coor_y_min[quad] ) m_coor_y_min[quad] = coor_y;
               if ( coor_y > m_coor_y_max[quad] ) m_coor_y_max[quad] = coor_y;
            }
            }
}

//--------------

void PixCoordsQuad::fillOneSectionTiltedInQuad(uint32_t quad, uint32_t sect, float xcenter, float ycenter, float zcenter, float rotation, float tilt)
{
  //cout << "PixCoordsQuad::fillOneSectionInQuad: " << endl;

    PixCoords2x1::ORIENTATION orient = PixCoords2x1::getOrientation(rotation);
  
    float xhalf  = m_pix_coords_2x1->getXCenterOffset_um(orient);
    float yhalf  = m_pix_coords_2x1->getYCenterOffset_um(orient);

    float tiltrad = tilt*m_degToRad;

    float sintilt = sin(tiltrad);
    float costilt = cos(tiltrad);

    // Calculate the corner-center offset due to the tilt

    float radius = sqrt(xhalf*xhalf + yhalf*yhalf);
    float sinPhi = yhalf / radius;
    float cosPhi = xhalf / radius;

    float rdPhi  = radius * tiltrad; // fabs(tiltrad) ?
    float dxOff  =  rdPhi * sinPhi;
    float dyOff  = -rdPhi * cosPhi;

    float xmin   = xcenter - xhalf + dxOff;
    float ymin   = ycenter - yhalf + dyOff;

    /*
    cout << " sect="      << sect 
         << " rotation="  << rotation
         << " tilt="      << tilt
         << " sin(tilt)=" << sintilt
         << " cos(tilt)=" << costilt
         << endl;
    */

    for (uint32_t col=0; col<NCols2x1; col++) {
    for (uint32_t row=0; row<NRows2x1; row++) {

       float x_in_sec = m_pix_coords_2x1->getPixCoorRotN90_um (orient, XCOOR, row, col);
       float y_in_sec = m_pix_coords_2x1->getPixCoorRotN90_um (orient, YCOOR, row, col);

       float x_tilted =  x_in_sec * costilt - y_in_sec * sintilt;
       float y_tilted =  x_in_sec * sintilt + y_in_sec * costilt; 

       float coor_x = xmin + x_tilted;
       float coor_y = ymin + y_tilted;

       m_coor_x[quad][sect][col][row] = coor_x;
       m_coor_y[quad][sect][col][row] = coor_y;

       if ( coor_x < m_coor_x_min[quad] ) m_coor_x_min[quad] = coor_x;
       if ( coor_x > m_coor_x_max[quad] ) m_coor_x_max[quad] = coor_x;
       if ( coor_y < m_coor_y_min[quad] ) m_coor_y_min[quad] = coor_y;
       if ( coor_y > m_coor_y_max[quad] ) m_coor_y_max[quad] = coor_y;
    }
    }
}

//--------------

void PixCoordsQuad::setConstXYMinMax()
{
        float pixSize_um = PSCalib::CSPadCalibPars::getRowSize_um();

	    for (uint32_t q=0; q<NQuadsInCSPad; q++) {
	      m_coor_x_min[q] = 0;
	      m_coor_y_min[q] = 0;
	      m_coor_x_max[q] = 850*pixSize_um;
	      m_coor_y_max[q] = 850*pixSize_um;
	    }
}

//--------------

float PixCoordsQuad::getPixCoorRot000_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return m_coor_x[quad][sect][col][row] - m_coor_x_min[quad];
    case CSPadPixCoords::PixCoords2x1::Y : return m_coor_y[quad][sect][col][row] - m_coor_y_min[quad];
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

float PixCoordsQuad::getPixCoorRot090_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return m_coor_y_max[quad] - m_coor_y[quad][sect][col][row];
    case CSPadPixCoords::PixCoords2x1::Y : return m_coor_x[quad][sect][col][row] - m_coor_x_min[quad];
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

float PixCoordsQuad::getPixCoorRot180_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return m_coor_x_max[quad] - m_coor_x[quad][sect][col][row];
    case CSPadPixCoords::PixCoords2x1::Y : return m_coor_y_max[quad] - m_coor_y[quad][sect][col][row];
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

float PixCoordsQuad::getPixCoorRot270_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return m_coor_y[quad][sect][col][row] - m_coor_y_min[quad];
    case CSPadPixCoords::PixCoords2x1::Y : return m_coor_x_max[quad] - m_coor_x[quad][sect][col][row];
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------
//--------------
//--------------
//--------------

float PixCoordsQuad::getPixCoorRot000_pix (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return getPixCoorRot000_um (icoor, quad, sect, row, col) * PSCalib::CSPadCalibPars::getRowUmToPix();
    case CSPadPixCoords::PixCoords2x1::Y : return getPixCoorRot000_um (icoor, quad, sect, row, col) * PSCalib::CSPadCalibPars::getColUmToPix();
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

float PixCoordsQuad::getPixCoorRot090_pix (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return getPixCoorRot090_um (icoor, quad, sect, row, col) * PSCalib::CSPadCalibPars::getColUmToPix();
    case CSPadPixCoords::PixCoords2x1::Y : return getPixCoorRot090_um (icoor, quad, sect, row, col) * PSCalib::CSPadCalibPars::getRowUmToPix();
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

float PixCoordsQuad::getPixCoorRot180_pix (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return getPixCoorRot180_um (icoor, quad, sect, row, col) * PSCalib::CSPadCalibPars::getRowUmToPix();
    case CSPadPixCoords::PixCoords2x1::Y : return getPixCoorRot180_um (icoor, quad, sect, row, col) * PSCalib::CSPadCalibPars::getColUmToPix();
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

float PixCoordsQuad::getPixCoorRot270_pix (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return getPixCoorRot270_um (icoor, quad, sect, row, col) * PSCalib::CSPadCalibPars::getColUmToPix();
    case CSPadPixCoords::PixCoords2x1::Y : return getPixCoorRot270_um (icoor, quad, sect, row, col) * PSCalib::CSPadCalibPars::getRowUmToPix();
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------

float PixCoordsQuad::getPixCoorRotN90_um (CSPadPixCoords::PixCoords2x1::ORIENTATION n90, 
                                          CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (n90)
    {
    case CSPadPixCoords::PixCoords2x1::R000 : return getPixCoorRot000_um (icoor, quad, sect, row, col);
    case CSPadPixCoords::PixCoords2x1::R090 : return getPixCoorRot090_um (icoor, quad, sect, row, col);
    case CSPadPixCoords::PixCoords2x1::R180 : return getPixCoorRot180_um (icoor, quad, sect, row, col);
    case CSPadPixCoords::PixCoords2x1::R270 : return getPixCoorRot270_um (icoor, quad, sect, row, col);
    default   : return 0;
    }
}

//--------------

float PixCoordsQuad::getPixCoorRotN90_pix(CSPadPixCoords::PixCoords2x1::ORIENTATION n90, 
                                          CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (n90)
    {
    case CSPadPixCoords::PixCoords2x1::R000 : return getPixCoorRot000_pix (icoor, quad, sect, row, col);
    case CSPadPixCoords::PixCoords2x1::R090 : return getPixCoorRot090_pix (icoor, quad, sect, row, col);
    case CSPadPixCoords::PixCoords2x1::R180 : return getPixCoorRot180_pix (icoor, quad, sect, row, col);
    case CSPadPixCoords::PixCoords2x1::R270 : return getPixCoorRot270_pix (icoor, quad, sect, row, col);
    default   : return 0;
    }
}

//--------------

float PixCoordsQuad::getPixCoorRotN90 ( CSPadPixCoords::PixCoords2x1::UNITS units, 
                                        CSPadPixCoords::PixCoords2x1::ORIENTATION n90, 
                                        CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned quad, unsigned sect, unsigned row, unsigned col)
{
  switch (units)
    {
    case  CSPadPixCoords::PixCoords2x1::UM  : return getPixCoorRotN90_um  (n90, icoor, quad, sect, row, col);
    case  CSPadPixCoords::PixCoords2x1::PIX : return getPixCoorRotN90_pix (n90, icoor, quad, sect, row, col);
    default   : return 0;
    }
}

//--------------
// Destructor --
//--------------

PixCoordsQuad::~PixCoordsQuad ()
{
}

} // namespace CSPadPixCoords
