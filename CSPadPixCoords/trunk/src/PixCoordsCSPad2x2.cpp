//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsCSPad2x2...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/PixCoordsCSPad2x2.h"
#include "MsgLogger/MsgLogger.h"

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

const char logger[] = "PixCoordsCSPad2x2";

//----------------
// Constructors --
//----------------
  PixCoordsCSPad2x2::PixCoordsCSPad2x2 (PixCoords2x1 *pix_coords_2x1, bool tiltIsApplied)
  : m_pix_coords_2x1(pix_coords_2x1)
  , m_cspad2x2_calibpars(0)
  , m_tiltIsApplied (tiltIsApplied)
{
  XCOOR = CSPadPixCoords::PixCoords2x1::X;
  YCOOR = CSPadPixCoords::PixCoords2x1::Y;
  ZCOOR = CSPadPixCoords::PixCoords2x1::Z;

  fillPixelCoordinateArrays();
  resetXYOriginAndMinMax();
  printXYLimits();
}

//--------------

PixCoordsCSPad2x2::PixCoordsCSPad2x2 (PixCoords2x1 *pix_coords_2x1,  PSCalib::CSPad2x2CalibPars *cspad2x2_calibpars, bool tiltIsApplied)
  : m_pix_coords_2x1(pix_coords_2x1)
  , m_cspad2x2_calibpars(cspad2x2_calibpars)
  , m_tiltIsApplied (tiltIsApplied)
{
  XCOOR = CSPadPixCoords::PixCoords2x1::X;
  YCOOR = CSPadPixCoords::PixCoords2x1::Y;
  ZCOOR = CSPadPixCoords::PixCoords2x1::Z;

  fillPixelCoordinateArrays();
  resetXYOriginAndMinMax();
  printXYLimits();
}

//--------------

void PixCoordsCSPad2x2::fillPixelCoordinateArrays()
{
        m_coor_x_min = 1e5;
        m_coor_x_max = 0;
        m_coor_y_min = 1e5;
        m_coor_y_max = 0;

        m_degToRad = 3.14159265359 / 180.; 

	// Approximate geometry for constructor w/o alignment parameters, where *cspad2x2_calibpars = 0;
	double xcenter_pix [] = {200,200};
	double ycenter_pix [] = { 95,308};
	double zcenter_pix [] = {  0,  0};
	double tilt_2x1    [] = {  0,  0};
	double rotation_2x1[] = {180,180}; // ...Just because of conventions in this code...

        double pixSize_um = PSCalib::CSPad2x2CalibPars::getRowSize_um();
	double xcenter, ycenter, zcenter, rotation, tilt;

        for (uint32_t sect=0; sect < N2x1InDet; ++sect)
          {
	    if (m_cspad2x2_calibpars) {
              xcenter  = m_cspad2x2_calibpars -> getCenterX (sect);
              ycenter  = m_cspad2x2_calibpars -> getCenterY (sect);
              zcenter  = m_cspad2x2_calibpars -> getCenterZ (sect);
              tilt     = m_cspad2x2_calibpars -> getTilt    (sect);
	    } else {
              xcenter  = xcenter_pix [sect];
              ycenter  = ycenter_pix [sect];
              zcenter  = zcenter_pix [sect];
              tilt     = tilt_2x1    [sect];
	    }

            xcenter *= pixSize_um;
            ycenter *= pixSize_um;
            rotation = rotation_2x1[sect];

            if (m_tiltIsApplied) fillOneSectionTiltedInDet(sect, xcenter, ycenter, zcenter, rotation, tilt);
            else                 fillOneSectionInDet      (sect, xcenter, ycenter, zcenter, rotation);

	    //cout << " sect=" << sect;
          }
	//cout << endl;
}

//--------------

void PixCoordsCSPad2x2::fillOneSectionInDet(uint32_t sect, double xcenter, double ycenter, double zcenter, double rotation)
{
            PixCoords2x1::ORIENTATION orient = PixCoords2x1::getOrientation(rotation);

            double xmin = xcenter - m_pix_coords_2x1->getXCenterOffset_um(orient);
            double ymin = ycenter - m_pix_coords_2x1->getYCenterOffset_um(orient);

            for (uint32_t col=0; col<NCols2x1; col++) {
            for (uint32_t row=0; row<NRows2x1; row++) {

	       double coor_x = xmin + m_pix_coords_2x1->getPixCoorRotN90_um (orient, XCOOR, row, col);
	       double coor_y = ymin + m_pix_coords_2x1->getPixCoorRotN90_um (orient, YCOOR, row, col);
	       m_coor_x[col][row][sect] = coor_x;
	       m_coor_y[col][row][sect] = coor_y;

               if ( coor_x < m_coor_x_min ) m_coor_x_min = coor_x;
               if ( coor_x > m_coor_x_max ) m_coor_x_max = coor_x;
               if ( coor_y < m_coor_y_min ) m_coor_y_min = coor_y;
               if ( coor_y > m_coor_y_max ) m_coor_y_max = coor_y;
            }
            }
}

//--------------

void PixCoordsCSPad2x2::fillOneSectionTiltedInDet(uint32_t sect, double xcenter, double ycenter, double zcenter, double rotation, double tilt)
{
  //cout << "PixCoordsCSPad2x2::fillOneSectionInQuad: " << endl;

    PixCoords2x1::ORIENTATION orient = PixCoords2x1::getOrientation(rotation);
  
    double xhalf  = m_pix_coords_2x1->getXCenterOffset_um(orient);
    double yhalf  = m_pix_coords_2x1->getYCenterOffset_um(orient);

    double tiltrad = tilt*m_degToRad;

    double sintilt = sin(tiltrad);
    double costilt = cos(tiltrad);

    // Calculate the corner-center offset due to the tilt

    double radius = sqrt(xhalf*xhalf + yhalf*yhalf);
    double sinPhi = yhalf / radius;
    double cosPhi = xhalf / radius;

    double rdPhi  = radius * tiltrad; // fabs(tiltrad) ?
    double dxOff  =  rdPhi * sinPhi;
    double dyOff  = -rdPhi * cosPhi;

    double xmin   = xcenter - xhalf + dxOff;
    double ymin   = ycenter - yhalf + dyOff;

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

       double x_in_sec = m_pix_coords_2x1->getPixCoorRotN90_um (orient, XCOOR, row, col);
       double y_in_sec = m_pix_coords_2x1->getPixCoorRotN90_um (orient, YCOOR, row, col);

       double x_tilted =  x_in_sec * costilt - y_in_sec * sintilt;
       double y_tilted =  x_in_sec * sintilt + y_in_sec * costilt; 

       double coor_x = xmin + x_tilted;
       double coor_y = ymin + y_tilted;

       m_coor_x[col][row][sect] = coor_x;
       m_coor_y[col][row][sect] = coor_y;

       if ( coor_x < m_coor_x_min ) m_coor_x_min = coor_x;
       if ( coor_x > m_coor_x_max ) m_coor_x_max = coor_x;
       if ( coor_y < m_coor_y_min ) m_coor_y_min = coor_y;
       if ( coor_y > m_coor_y_max ) m_coor_y_max = coor_y;
    }
    }
}

//--------------

void PixCoordsCSPad2x2::resetXYOriginAndMinMax()
{
    for (uint32_t col=0; col<NCols2x1;  col++) {
    for (uint32_t row=0; row<NRows2x1;  row++) {
    for (uint32_t sec=0; sec<N2x1InDet; sec++) {

       m_coor_x[col][row][sec] -= m_coor_x_min;
       m_coor_y[col][row][sec] -= m_coor_y_min;
    }
    }
    }

    //double pixSize_um = PSCalib::CSPad2x2CalibPars::getRowSize_um();
    //m_coor_x_max = 400*pixSize_um;
    //m_coor_y_max = 400*pixSize_um;
    m_coor_x_max -= m_coor_x_min;
    m_coor_y_max -= m_coor_y_min;
    m_coor_x_min = 0;
    m_coor_y_min = 0;
}

//--------------

void PixCoordsCSPad2x2::printXYLimits()
{
  MsgLog(logger, info, "  Xmin: " << m_coor_x_min << "  Xmax: " << m_coor_x_max
	            << "  Ymin: " << m_coor_y_min << "  Ymax: " << m_coor_y_max );
}

//--------------

double PixCoordsCSPad2x2::getPixCoor_um (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return m_coor_x[col][row][sect];
    case CSPadPixCoords::PixCoords2x1::Y : return m_coor_y[col][row][sect];
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------
//--------------
//--------------
//--------------

double PixCoordsCSPad2x2::getPixCoor_pix (CSPadPixCoords::PixCoords2x1::COORDINATE icoor, unsigned sect, unsigned row, unsigned col)
{
  switch (icoor)
    {
    case CSPadPixCoords::PixCoords2x1::X : return getPixCoor_um (icoor, sect, row, col) * PSCalib::CSPad2x2CalibPars::getRowUmToPix();
    case CSPadPixCoords::PixCoords2x1::Y : return getPixCoor_um (icoor, sect, row, col) * PSCalib::CSPad2x2CalibPars::getColUmToPix();
    case CSPadPixCoords::PixCoords2x1::Z : return 0;
    default: return 0;
    }
}

//--------------
// Destructor --
//--------------

PixCoordsCSPad2x2::~PixCoordsCSPad2x2 ()
{
}

} // namespace CSPadPixCoords
