//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PixCoords2x1...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSCalib/PixCoords2x1V2.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <math.h>      // sin, cos
//#include <cmath> 
//#include <algorithm> // std::copy
#include <iostream>    // cout
//#include <fstream>
//#include <string>
using namespace std;


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSCalib/CSPadCalibPars.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSCalib {

//----------------
// GLOBAL methods
//----------------


void rotation(const double* x, const double* y, unsigned size, double angle_deg, double* xrot, double* yrot)
{
    const double angle_rad = angle_deg * DEG_TO_RAD; // 3.14159265359 / 180; 
    const double C = cos(angle_rad);
    const double S = sin(angle_rad);
    rotation(x, y, size, C, S, xrot, yrot);
}

//--------------

void rotation(const double* x, const double* y, unsigned size, double C, double S, double* xrot, double* yrot)
{
  for (unsigned i=0; i<size; i++, xrot++, yrot++, *x++, *y++) {
    *xrot = *x *C - *y *S;
    *yrot = *y *C + *x *S; 
  }
}

//--------------

double min_of_array(const double* arr, unsigned size)
{
  double min=arr[0]; for(unsigned i=1; i<size; ++i) { if(arr[i] < min) min=arr[i]; } 
  return min;
}

//--------------

double max_of_array(const double* arr, unsigned size)
{
  double max=arr[0]; for(unsigned i=1; i<size; ++i) { if(arr[i] > max) max=arr[i]; } 
  return max;
}


//----------------
// Constructors --
//----------------

PixCoords2x1V2::PixCoords2x1V2 (bool use_wide_pix_center)
{
  //cout << "C-tor of PixCoords2x1V2" << endl;

  m_use_wide_pix_center = use_wide_pix_center;
  m_angle_deg = 123456;

  make_maps_of_2x1_pix_coordinates();
}

//--------------
// Destructor --
//--------------

PixCoords2x1V2::~PixCoords2x1V2 ()
{
}


//--------------

const unsigned PixCoords2x1V2::IND_CORNER[NCORNERS] = {0, COLS2X1-1, (ROWS2X1-1)*COLS2X1, ROWS2X1*COLS2X1-1};

//--------------

void PixCoords2x1V2::make_maps_of_2x1_pix_coordinates()
{
  // Define x-coordinate of pixels
  double x_offset = PIX_SIZE_WIDE - PIX_SIZE_COLS / 2;
  for (unsigned c=0; c<COLS2X1HALF; c++) m_x_rhs[c] = c * PIX_SIZE_COLS + x_offset;
  if (m_use_wide_pix_center)        m_x_rhs[0] = PIX_SIZE_WIDE / 2;
  for (unsigned c=0; c<COLS2X1HALF; c++) { m_x_arr_um[c]               = -m_x_rhs[COLS2X1HALF-1-c];
                                      m_x_arr_um[COLS2X1HALF + c] =  m_x_rhs[c]; }

  // Define y-coordinate of pixels
  double y_offset = (ROWS2X1-1) * PIX_SIZE_ROWS / 2;
  for (unsigned r=0; r<ROWS2X1; r++) m_y_arr_um[r] = y_offset - r * PIX_SIZE_ROWS;

  for (unsigned c=0; c<COLS2X1; c++) { m_x_arr_pix[c] = m_x_arr_um[c] / PIX_SIZE_COLS; }
  for (unsigned r=0; r<ROWS2X1; r++) { m_y_arr_pix[r] = m_y_arr_um[r] / PIX_SIZE_ROWS; }

  for (unsigned r=0; r<ROWS2X1; r++) {
    for (unsigned c=0; c<COLS2X1; c++) {
      m_x_map_2x1_um [r][c] = m_x_arr_um [c];
      m_y_map_2x1_um [r][c] = m_y_arr_um [r];
      m_x_map_2x1_pix[r][c] = m_x_arr_pix[c];
      m_y_map_2x1_pix[r][c] = m_y_arr_pix[r];
    }
  }

  std::fill_n(&m_z_map_2x1[0][0], int(SIZE2X1), double(0));
}

//--------------

void PixCoords2x1V2::print_member_data()
{
  cout << "PixCoords2x1V2::print_member_data():"       
       << "\n ROWS2X1               " << ROWS2X1       
       << "\n COLS2X1               " << COLS2X1       
       << "\n COLS2X1HALF           " << COLS2X1HALF   
       << "\n PIX_SIZE_COLS         " << PIX_SIZE_COLS 
       << "\n PIX_SIZE_ROWS         " << PIX_SIZE_ROWS 
       << "\n PIX_SIZE_WIDE         " << PIX_SIZE_WIDE    
       << "\n PIX_SIZE_UM           " << PIX_SIZE_UM
       << "\n UM_TO_PIX             " << UM_TO_PIX
       << "\n m_use_wide_pix_center " << m_use_wide_pix_center 
       << "\n m_angle_deg           " << m_angle_deg 
    //<< "\n        " <<     
       << "\n";
}

//--------------

void PixCoords2x1V2::print_map_min_max(UNITS units, const double& angle_deg)
{
  cout << "  2x1 coordinate map limits for units: " << units << " and angle(deg): " << angle_deg << "\n";
  cout << "  xmin =  " << get_min_of_coord_map_2x1 (AXIS_X, units, angle_deg) << "\n";
  cout << "  xmax =  " << get_max_of_coord_map_2x1 (AXIS_X, units, angle_deg) << "\n";
  cout << "  ymin =  " << get_min_of_coord_map_2x1 (AXIS_Y, units, angle_deg) << "\n";
  cout << "  ymax =  " << get_max_of_coord_map_2x1 (AXIS_Y, units, angle_deg) << "\n";
  cout << "  zmin =  " << get_min_of_coord_map_2x1 (AXIS_Z, units, angle_deg) << "\n";
  cout << "  zmax =  " << get_max_of_coord_map_2x1 (AXIS_Z, units, angle_deg) << "\n";
}

//--------------

void PixCoords2x1V2::print_coord_arrs_2x1()
{
  cout << "\nPixCoords2x1V2::print_coord_arrs_2x1\n";

  cout << "m_x_arr_pix:\n"; 
  for (unsigned counter=0, c=0; c<COLS2X1; c++) {
    cout << " " << m_x_arr_pix[c];
    if (++counter > 19) { counter=0; cout << "\n"; }
  }
  cout << "\n"; 

  cout << "m_y_arr_pix:\n"; 
  for (unsigned counter=0, r=0; r<ROWS2X1; r++) { 
    cout << " " << m_y_arr_pix[r];
    if (++counter > 19) { counter=0; cout << "\n"; }
  }
  cout << "\n"; 
}

//--------------

double* PixCoords2x1V2::get_coord_map_2x1 (AXIS axis, UNITS units, const double& angle_deg) 
{ 
  if(axis == AXIS_Z) return &m_z_map_2x1[0][0];

  if(angle_deg != m_angle_deg or units != m_units) {
    m_angle_deg = angle_deg;
    m_units     = units;

    const double* x;
    const double* y;
    
    switch (units)
      {
      case PIX : 
        x = &m_x_map_2x1_pix[0][0];
        y = &m_y_map_2x1_pix[0][0];
        break;

      default : // case UM : 
        x = &m_x_map_2x1_um [0][0];
        y = &m_y_map_2x1_um [0][0];
      }    
      PSCalib::rotation(x, y, SIZE2X1, angle_deg, &m_x_map_2x1_rot[0][0], &m_y_map_2x1_rot[0][0]); 
  }

  switch (axis)
    {
      case AXIS_X : return &m_x_map_2x1_rot [0][0];
      case AXIS_Y : return &m_y_map_2x1_rot [0][0];
      default     : return &m_x_map_2x1_rot [0][0];
    }
} 

//--------------

double PixCoords2x1V2::get_min_of_coord_map_2x1 (AXIS axis, UNITS units, const double& angle_deg) 
{ 
  double* arr = get_coord_map_2x1 (axis, units, angle_deg);
  double corner_coords[NCORNERS];
  for (unsigned i=0; i<NCORNERS; ++i) corner_coords[i] = arr[IND_CORNER[i]];
  return PSCalib::min_of_array(corner_coords, NCORNERS); 
}

//--------------

double PixCoords2x1V2::get_max_of_coord_map_2x1 (AXIS axis, UNITS units, const double& angle_deg) 
{ 
  double* arr = get_coord_map_2x1 (axis, units, angle_deg);
  double corner_coords[NCORNERS];
  for (unsigned i=0; i<NCORNERS; ++i) corner_coords[i] = arr[IND_CORNER[i]];
  return PSCalib::max_of_array(corner_coords, NCORNERS); 
}

//--------------
//--------------
//--------------
//--------------

} // namespace PSCalib

//--------------
