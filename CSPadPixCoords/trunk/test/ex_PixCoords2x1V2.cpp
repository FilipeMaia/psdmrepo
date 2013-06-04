//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class CSPadCalibPars of the PSCalib packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "CSPadPixCoords/PixCoords2x1V2.h"
#include "CSPadPixCoords/Image2D.h"

#include <string>
#include <iostream>

//using std::cout;
//using std::endl;

using namespace std;

typedef CSPadPixCoords::PixCoords2x1V2 PC2X1;

//-----------------

void test01()
{
  PC2X1 *pix_coords_2x1 = new PC2X1();  

  pix_coords_2x1 -> print_member_data();
  pix_coords_2x1 -> print_coord_arrs_2x1();
  //pix_coords_2x1 -> print_xy_corners();
}

//-----------------

void test02()
{
  PC2X1 *pix_coords_2x1 = new PC2X1();  

  double* x_arr = pix_coords_2x1 -> get_x_map_2x1_pix ();
  double* y_arr = pix_coords_2x1 -> get_y_map_2x1_pix ();

  CSPadPixCoords::Image2D<double> *img2d_x = new CSPadPixCoords::Image2D<double>(x_arr, PC2X1::ROWS2X1, PC2X1::COLS2X1);
  CSPadPixCoords::Image2D<double> *img2d_y = new CSPadPixCoords::Image2D<double>(y_arr, PC2X1::ROWS2X1, PC2X1::COLS2X1);

  img2d_x  -> saveImageInFile("test-x-array.txt",0);
  img2d_y  -> saveImageInFile("test-y-array.txt",0);
}


//-----------------

void test03()
{
  enum{ NX=500, NY=500 };
  double img_arr[NY][NX];
  std::fill_n(&img_arr[0][0], int(NX*NY), double(0));

  PC2X1 *pix_coords_2x1 = new PC2X1(true);  

  // Assignment to coordinates
  unsigned NROWS = PC2X1::ROWS2X1;
  unsigned NCOLS = PC2X1::COLS2X1;
  double   PSIZE = 109.92;

  double angle = 5;

  pix_coords_2x1 -> print_map_min_max(PC2X1::PIX, 0);
  pix_coords_2x1 -> print_map_min_max(PC2X1::PIX, angle);
  pix_coords_2x1 -> print_map_min_max(PC2X1::UM, 0);
  pix_coords_2x1 -> print_map_min_max(PC2X1::UM, angle);

  //double* x_arr = pix_coords_2x1 -> get_x_map_2x1_um ();
  //double* y_arr = pix_coords_2x1 -> get_y_map_2x1_um ();

  double* x_arr = pix_coords_2x1 -> get_coord_map_2x1 (PC2X1::X, PC2X1::UM, angle);
  double* y_arr = pix_coords_2x1 -> get_coord_map_2x1 (PC2X1::Y, PC2X1::UM, angle);

  for (unsigned r=0; r<NROWS; r++){
  for (unsigned c=0; c<NCOLS; c++){

    int ix = 200 + (int)(x_arr[r*NCOLS + c] / PSIZE);
    int iy = 200 + (int)(y_arr[r*NCOLS + c] / PSIZE);

    img_arr[ix][iy] = ix;
  }
  }

  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(&img_arr[0][0],NY,NX);
  img2d -> saveImageInFile("test-img.txt",0);
}

//-----------------

int main ()
{
  test01();
  //test02();
  test03();

  return 0;
}

//-----------------

