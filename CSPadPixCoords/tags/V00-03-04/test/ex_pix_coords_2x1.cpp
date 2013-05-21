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

#include "CSPadPixCoords/PixCoords2x1.h"
#include "CSPadPixCoords/Image2D.h"

#include <string>
#include <iostream>

using std::cout;
using std::endl;

//using namespace std;

//-----------------

void test01()
{
  CSPadPixCoords::PixCoords2x1 *pix_coords_2x1 = new CSPadPixCoords::PixCoords2x1();  

  pix_coords_2x1 -> print_member_data();

  pix_coords_2x1 -> print_selected_coords_2x1(CSPadPixCoords::PixCoords2x1::ROW);
  pix_coords_2x1 -> print_selected_coords_2x1(CSPadPixCoords::PixCoords2x1::COL);
  pix_coords_2x1 -> print_selected_coords_2x1(CSPadPixCoords::PixCoords2x1::ORT);
  //pix_coords_2x1 -> print_selected_coords_2x1(-1); // works as for ROW
}

//-----------------

void test02()
{
  enum{ NX=600, NY=300 };
  double arr[NY][NX];

  for (unsigned iy=0; iy<NY; iy++){
  for (unsigned ix=0; ix<NX; ix++){
    arr[iy][ix] = ix + iy;
  }
  }

  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(&arr[0][0],NY,NX);
  img2d -> saveImageInFile("test.txt",0);
}

//-----------------

void test03()
{

  // Initialization
  enum{ NX=500, NY=500 };
  double arr[NY][NX];
  for (unsigned iy=0; iy<NY; iy++){
  for (unsigned ix=0; ix<NX; ix++){
    arr[iy][ix] = 0;
  }
  }


  CSPadPixCoords::PixCoords2x1 *pix_coords_2x1 = new CSPadPixCoords::PixCoords2x1();  
  pix_coords_2x1 -> print_member_data();

  // Assignment to coordinates
  enum{ NROWS=CSPadPixCoords::PixCoords2x1::NRows2x1, 
        NCOLS=CSPadPixCoords::PixCoords2x1::NCols2x1 };

  CSPadPixCoords::PixCoords2x1::ORIENTATION rot=CSPadPixCoords::PixCoords2x1::R000;
  CSPadPixCoords::PixCoords2x1::COORDINATE  X  =CSPadPixCoords::PixCoords2x1::X;
  CSPadPixCoords::PixCoords2x1::COORDINATE  Y  =CSPadPixCoords::PixCoords2x1::Y;

  cout << "NROWS=" << NROWS << endl;
  cout << "NCOLS=" << NCOLS << endl;



  //double um_to_pixels = 1./109.92;
  //unsigned c=100;

  unsigned mrgx=20;
  unsigned mrgy=20;

  for (unsigned r=0; r<NROWS; r++){
  for (unsigned c=0; c<NCOLS; c++){

    double x = pix_coords_2x1 -> getPixCoorRotN90_pix (rot, X, r, c);
    double y = pix_coords_2x1 -> getPixCoorRotN90_pix (rot, Y, r, c);

    //int iy = mrgy + (int)(y*um_to_pixels);
    //int ix = mrgx + (int)(x*um_to_pixels);

    int iy = mrgy + (int)y;
    int ix = mrgx + (int)x;

    /*
    cout << "  ix=" << ix 
         << "  iy=" << iy
         << endl;
    */   
    arr[iy][ix] = ix;
  }
  }

  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(&arr[0][0],NY,NX);
  img2d -> saveImageInFile("test.txt",0);
}

//-----------------

int main ()
{
  test01();
  test02();
  test03();

  return 0;
}

//-----------------

