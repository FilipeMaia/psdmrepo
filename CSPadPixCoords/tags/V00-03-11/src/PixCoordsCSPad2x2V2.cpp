//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsCSPad2x2V2...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/PixCoordsCSPad2x2V2.h"
//#include "MsgLogger/MsgLogger.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <math.h>
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include <boost/math/constants/constants.hpp>

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;

namespace CSPadPixCoords {

const char logger[] = "CSPadPixCoords";

//----------------
// Constructors --
//----------------

/*
PixCoordsCSPad2x2V2::PixCoordsCSPad2x2V2 ()
    : PixCoords2x1V2 ()
    , m_tiltIsApplied (false)
{
  // Use default calibration parameters
  m_cspad2x2_calibpars = new PSCalib::CSPad2x2CalibPars(); 
  fillPixelCoordinateArrays();
  resetXYOriginAndMinMax();
}
*/

//--------------

PixCoordsCSPad2x2V2::PixCoordsCSPad2x2V2 (PSCalib::CSPad2x2CalibPars *cspad2x2_calibpars, bool tiltIsApplied, bool use_wide_pix_center)
    : PixCoords2x1V2(use_wide_pix_center)
    , m_cspad2x2_calibpars(cspad2x2_calibpars)
    , m_tiltIsApplied (tiltIsApplied)
{
  fillPixelCoordinateArrays();
  resetXYOriginAndMinMax();
  //printXYLimits();
}

//--------------

void PixCoordsCSPad2x2V2::fillPixelCoordinateArrays()
{
    m_coor_x_min = 1e5;
    m_coor_x_max = 0;
    m_coor_y_min = 1e5;
    m_coor_y_max = 0;

    double rotation_2x1[] = {180,180}; // ...Just because of conventions in this code...

    double xcenter, ycenter, zcenter, rotation, tilt;

    for (unsigned sect=0; sect < N2X1_IN_DET; ++sect)
      {
        xcenter  = m_cspad2x2_calibpars -> getCenterX (sect) * PIX_SIZE_UM;
        ycenter  = m_cspad2x2_calibpars -> getCenterY (sect) * PIX_SIZE_UM;
        zcenter  = m_cspad2x2_calibpars -> getCenterZ (sect);
        tilt     = m_cspad2x2_calibpars -> getTilt    (sect);

        rotation = rotation_2x1[sect];
        if (m_tiltIsApplied) rotation += tilt;

        fillOneSectionInDet (sect, xcenter, ycenter, zcenter, rotation);
      }
}

//--------------

void PixCoordsCSPad2x2V2::fillOneSectionInDet(uint32_t sect, double xcenter, double ycenter, double zcenter, double rotation)
{
    double* x_map_arr = get_coord_map_2x1 (AXIS_X, UM, rotation); 
    double* y_map_arr = get_coord_map_2x1 (AXIS_Y, UM, rotation); 

    for (unsigned row=0; row<ROWS2X1; row++) {
    for (unsigned col=0; col<COLS2X1; col++) {

       unsigned ind = row*COLS2X1 + col;

       double coor_x = xcenter + x_map_arr[ind];
       double coor_y = ycenter + y_map_arr[ind]; 

       m_coor_x[row][col][sect] = coor_x;
       m_coor_y[row][col][sect] = coor_y;

       if ( coor_x < m_coor_x_min ) m_coor_x_min = coor_x;
       if ( coor_x > m_coor_x_max ) m_coor_x_max = coor_x;
       if ( coor_y < m_coor_y_min ) m_coor_y_min = coor_y;
       if ( coor_y > m_coor_y_max ) m_coor_y_max = coor_y;
    }
    }
}

//--------------

void PixCoordsCSPad2x2V2::resetXYOriginAndMinMax()
{
    for (unsigned row=0; row<ROWS2X1;  row++) {
    for (unsigned col=0; col<COLS2X1;  col++) {
    for (unsigned sec=0; sec<N2X1_IN_DET; sec++) {

       m_coor_x[row][col][sec] -= m_coor_x_min;
       m_coor_y[row][col][sec] -= m_coor_y_min;
    }
    }
    }

    m_coor_x_max -= m_coor_x_min;
    m_coor_y_max -= m_coor_y_min;
    m_coor_x_min = 0;
    m_coor_y_min = 0;
}

//--------------

double PixCoordsCSPad2x2V2::getPixCoor_um (AXIS axis, unsigned sect, unsigned row, unsigned col)
{
  //cout << "      get sect:" << sect << "  r:" << row << "  c:" << col << "  axis:" << axis << "\n"; 
  switch (axis)
    {
    case AXIS_X : return m_coor_x[row][col][sect];
    case AXIS_Y : return m_coor_y[row][col][sect];
    case AXIS_Z : return 0;
    default: return 0;
    }
}

//--------------

double PixCoordsCSPad2x2V2::getPixCoor_pix (AXIS axis, unsigned sect, unsigned row, unsigned col)
{
    return getPixCoor_um (axis, sect, row, col) * UM_TO_PIX;
}

//--------------

void PixCoordsCSPad2x2V2::printXYLimits()
{
  std::stringstream ss; ss << "  Xmin: " << m_coor_x_min 
	  		   << "  Xmax: " << m_coor_x_max
			   << "  Ymin: " << m_coor_y_min
                           << "  Ymax: " << m_coor_y_max;

  //MsgLog(logger, info, ss.str());
  cout << "PixCoordsCSPad2x2V2::printXYLimits():" << ss.str() << "\n";
}

//--------------

void PixCoordsCSPad2x2V2::printConstants()
{
  cout << "PixCoordsCSPad2x2V2::printConstants():"    
       << "\n  ROWS2X1       = " << ROWS2X1    
       << "\n  COLS2X1       = " << COLS2X1     
       << "\n  COLS2X1HALF   = " << COLS2X1HALF 
       << "\n  SIZE2X1       = " << SIZE2X1     
       << "\n  PIX_SIZE_COLS = " << PIX_SIZE_COLS 
       << "\n  PIX_SIZE_ROWS = " << PIX_SIZE_ROWS 
       << "\n  PIX_SIZE_WIDE = " << PIX_SIZE_WIDE 
       << "\n  PIX_SIZE_UM   = " << PIX_SIZE_UM   
       << "\n  UM_TO_PIX     = " << UM_TO_PIX     
       << "\n";
}

//--------------

void PixCoordsCSPad2x2V2::printCoordArray(unsigned r1, unsigned r2, unsigned c1, unsigned c2)
{
    cout << "PixCoordsCSPad2x2V2::printCoordArray():"
    	 << "\nsizeof(m_coor_x) / sizeof(double)=" << sizeof(m_coor_x) / sizeof(double)
    	 << "\nsizeof(m_coor_y) / sizeof(double)=" << sizeof(m_coor_y) / sizeof(double) 
         << "\n";

    for (unsigned row=r1; row<r2;  row++) {
      cout << "\nrow=" << row << ": ";
    for (unsigned col=c1; col<c2;  col++) {
    for (unsigned sec=0; sec<N2X1_IN_DET; sec++) {
      cout << " (" << m_coor_x[row][col][sec] << ", " << m_coor_y[row][col][sec] << ") ";
    }
    }
    }
    cout <<"\n";  
}


//--------------
// Destructor --
//--------------

PixCoordsCSPad2x2V2::~PixCoordsCSPad2x2V2 ()
{
}

} // namespace CSPadPixCoords
