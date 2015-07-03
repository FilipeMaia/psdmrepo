//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class SegGeometry...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSCalib/SegGeometry.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <math.h>      // sin, cos
//#include <cmath> 
//#include <algorithm> // std::copy
#include <iostream>    // cout
//#include <fstream>
//#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------



namespace PSCalib {

//----------------
// GLOBAL methods
//----------------

void rotation_ang(const double* x, const double* y, unsigned size, double angle_deg, double* xrot, double* yrot)
{
    const double angle_rad = angle_deg * DEG_TO_RAD;
    const double C = cos(angle_rad);
    const double S = sin(angle_rad);
    rotation_cs(x, y, size, C, S, xrot, yrot);
}

//--------------

void rotation_cs(const double* x, const double* y, unsigned size, double C, double S, double* xrot, double* yrot)
{
  for (unsigned i=0; i<size; i++, xrot++, yrot++, *x++, *y++) {
    *xrot = *x *C - *y *S;
    *yrot = *y *C + *x *S; 
  }
}

//--------------

double min_of_arr(const double* arr, unsigned size)
{
  double min=arr[0]; for(unsigned i=1; i<size; ++i) { if(arr[i] < min) min=arr[i]; } 
  return min;
}

//--------------

double max_of_arr(const double* arr, unsigned size)
{
  double max=arr[0]; for(unsigned i=1; i<size; ++i) { if(arr[i] > max) max=arr[i]; } 
  return max;
}

//----------------
// Constructors --
//----------------
//SegGeometry::SegGeometry (){}

//--------------
// Destructor --
//--------------
//SegGeometry::~SegGeometry (){}

//--------------
//--------------
//--------------
//--------------

} // namespace PSCalib

//--------------
