//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoordsCSPadV2...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/PixCoordsCSPadV2.h"
//#include "MsgLogger/MsgLogger.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <math.h>
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
//#include <boost/math/constants/constants.hpp>

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
PixCoordsCSPadV2::PixCoordsCSPadV2 ()
    : PixCoords2x1V2 ()
    , m_tiltIsApplied (false)
{
  // Use default calibration parameters
  m_cspad_calibpars  = new PSCalib::CSPadCalibPars(); 
  m_cspad_configpars = new CSPadConfigPars(); 
  fillPixelCoordinateArrays();
  resetXYOriginAndMinMax();
}
*/

//--------------

PixCoordsCSPadV2::PixCoordsCSPadV2 (PSCalib::CSPadCalibPars *cspad_calibpars, bool tiltIsApplied, bool use_wide_pix_center)
    : PixCoords2x1V2(use_wide_pix_center)
    , m_cspad_calibpars(cspad_calibpars)
    , m_tiltIsApplied(tiltIsApplied)
{
  fillPixelCoordinateArrays();
  resetXYOriginAndMinMax();
  //printXYLimits();
}

//--------------

void PixCoordsCSPadV2::fillPixelCoordinateArrays()
{
    m_coor_x_min = 1e5;
    m_coor_x_max = 0;
    m_coor_y_min = 1e5;
    m_coor_y_max = 0;


    // Orientation of 2x1s:        0   1     2     3     4     5     6     7
    double orient_2x1_in_quad[] = {0., 0., 270., 270., 180., 180., 270., 270.};

    // Orientation of quads:    0  1    2     3 
    double orient_of_quad[] = {90, 0, -90, -180};

    double xcenter, ycenter, zcenter, rotation, tilt;
    size_t q, s; 

    for (unsigned sect=0; sect < N2X1_IN_DET; ++sect)
      {
        q = sect / 8;
        s = sect % 8;

        xcenter  = m_cspad_calibpars -> getCenterGlobalX (q,s) * PIX_SIZE_UM;
        ycenter  = m_cspad_calibpars -> getCenterGlobalY (q,s) * PIX_SIZE_UM;
        zcenter  = m_cspad_calibpars -> getCenterGlobalZ (q,s) * PIX_SIZE_UM;
        tilt     = m_cspad_calibpars -> getTilt          (q,s);

        //cout << "sect:" << sect << "  quad:" << q << "  sect in quad:" << s << "\n"; // OK

        rotation = orient_of_quad[q] + orient_2x1_in_quad[s];

        if (m_tiltIsApplied) rotation += tilt;

        fillOneSectionInDet (sect, xcenter, ycenter, zcenter, rotation);
      }
}

//--------------

void PixCoordsCSPadV2::fillOneSectionInDet(uint32_t sect, double xcenter, double ycenter, double zcenter, double rotation)
{
    double* x_map_arr = get_coord_map_2x1 (AXIS_X, UM, rotation); 
    double* y_map_arr = get_coord_map_2x1 (AXIS_Y, UM, rotation); 

    for (unsigned row=0; row<ROWS2X1; row++) {
    for (unsigned col=0; col<COLS2X1; col++) {

       unsigned ind = row*COLS2X1 + col;

       double coor_x = xcenter + x_map_arr[ind];
       double coor_y = ycenter + y_map_arr[ind]; 

       m_coor_x[sect][row][col] = coor_x;
       m_coor_y[sect][row][col] = coor_y;
       m_coor_z[sect][row][col] = 0;

       if ( coor_x < m_coor_x_min ) m_coor_x_min = coor_x;
       if ( coor_x > m_coor_x_max ) m_coor_x_max = coor_x;
       if ( coor_y < m_coor_y_min ) m_coor_y_min = coor_y;
       if ( coor_y > m_coor_y_max ) m_coor_y_max = coor_y;
    }
    }
}

//--------------

void PixCoordsCSPadV2::resetXYOriginAndMinMax()
{
    for (unsigned sec=0; sec<N2X1_IN_DET; sec++) {
    for (unsigned row=0; row<ROWS2X1;  row++) {
    for (unsigned col=0; col<COLS2X1;  col++) {

       m_coor_x[sec][row][col] -= m_coor_x_min;
       m_coor_y[sec][row][col] -= m_coor_y_min;
    }
    }
    }

    m_coor_x_max -= m_coor_x_min;
    m_coor_y_max -= m_coor_y_min;
    m_coor_x_min = 0;
    m_coor_y_min = 0;
    m_coor_z_min = 0;
    m_coor_z_max = 0;
}

//--------------

double PixCoordsCSPadV2::getPixCoor_um (AXIS axis, unsigned sect, unsigned row, unsigned col)
{
  //cout << "      get sect:" << sect << "  r:" << row << "  c:" << col << "  axis:" << axis << "\n"; 
  switch (axis)
    {
    case AXIS_X : return m_coor_x[sect][row][col];
    case AXIS_Y : return m_coor_y[sect][row][col];
    case AXIS_Z : return m_coor_z[sect][row][col];
    default: return 0;
    }
}

//--------------

double PixCoordsCSPadV2::getPixCoor_pix (AXIS axis, unsigned sect, unsigned row, unsigned col)
{
    return getPixCoor_um (axis, sect, row, col) * UM_TO_PIX;
}

//--------------

double* PixCoordsCSPadV2::getPixCoorArr_um (AXIS axis)
{
  //cout << "  axis:" << axis << "\n"; 
  switch (axis)
    {
    case AXIS_X : return &m_coor_x[0][0][0];
    case AXIS_Y : return &m_coor_y[0][0][0];
    case AXIS_Z : return &m_coor_z[0][0][0];
    default     : return 0;
    }
}

//--------------

ndarray<double,3> PixCoordsCSPadV2::getPixCoorNDArrShapedAsData_um (AXIS axis, CSPadConfigPars *cspad_configpars)
{
  //cout << "  get pix coordinate ndarray for axis:" << axis << "\n"; 

  double *p_pix_arr=0;
      
  switch (axis)
    {
    case AXIS_X : { p_pix_arr = &m_coor_x[0][0][0]; break; }
    case AXIS_Y : { p_pix_arr = &m_coor_y[0][0][0]; break; } 
    case AXIS_Z : { p_pix_arr = &m_coor_z[0][0][0]; break;  }
    default     : { break; }
    }

  ndarray<double,3> nda = make_ndarray(p_pix_arr, N2X1_IN_DET, ROWS2X1, COLS2X1); 
  return cspad_configpars->getCSPadPixNDArrShapedAsData<double>( nda ); 
}

//--------------

void PixCoordsCSPadV2::printXYLimits()
{
  std::stringstream ss; ss << "  Xmin: " << m_coor_x_min 
	  		   << "  Xmax: " << m_coor_x_max
			   << "  Ymin: " << m_coor_y_min
                           << "  Ymax: " << m_coor_y_max
			   << "  Zmin: " << m_coor_z_min
                           << "  Zmax: " << m_coor_z_max;

  //MsgLog(logger, info, ss.str());
  cout << "PixCoordsCSPadV2::printXYLimits():" << ss.str() << "\n";
}

//--------------

void PixCoordsCSPadV2::printConstants()
{
  cout << "PixCoordsCSPadV2::printConstants():"    
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

void PixCoordsCSPadV2::printCoordArray(unsigned r1, unsigned r2, unsigned c1, unsigned c2)
{
    cout << "PixCoordsCSPadV2::printCoordArray():"
    	 << "\nsizeof(m_coor_x) / sizeof(double)=" << sizeof(m_coor_x) / sizeof(double)
    	 << "\nsizeof(m_coor_y) / sizeof(double)=" << sizeof(m_coor_y) / sizeof(double) 
         << "\n";

    for (unsigned sec=0; sec<N2X1_IN_DET; sec++) {
      cout << "\nsection=" << sec << ": ";
    for (unsigned row=r1; row<r2;  row++) {
      cout << "\nrow=" << row << ": ";
    for (unsigned col=c1; col<c2;  col++) {
      cout << " (" << m_coor_x[sec][row][col] << ", " << m_coor_y[sec][row][col] << ") ";
    }
    }
    }
    cout <<"\n";  
}


//--------------
// Destructor --
//--------------

PixCoordsCSPadV2::~PixCoordsCSPadV2 ()
{
}

} // namespace CSPadPixCoords
