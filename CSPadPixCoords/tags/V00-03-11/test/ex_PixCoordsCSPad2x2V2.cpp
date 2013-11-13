//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class PixCoordsCSPad2x2V2 of the CSPadPixCoords packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "CSPadPixCoords/PixCoordsCSPad2x2V2.h"
#include "CSPadPixCoords/Image2D.h"
#include "PSCalib/CSPad2x2CalibPars.h"

#include <string>
#include <iostream>

//using std::cout;
//using std::endl;

using namespace std;

//typedef CSPadPixCoords::PixCoords2x1V2 PC2X1;
typedef CSPadPixCoords::PixCoordsCSPad2x2V2 PC2X2;
typedef PSCalib::CSPad2x2CalibPars CALIB2X2;

//-----------------

void test01()
{
  PC2X2 *pix_coords_2x2 = new PC2X2();  

  pix_coords_2x2 -> printXYLimits();
}

//-----------------

void test02()
{
  const unsigned NX=420, NY=420;
  double img_arr[NY][NX];
  std::fill_n(&img_arr[0][0], int(NX*NY), double(0));

  PC2X2 *pix_coords_2x2 = new PC2X2();  
  pix_coords_2x2 -> printXYLimits();

  // Assignment to coordinates

  for (unsigned r=0; r<PC2X2::ROWS2X1; r++){
  for (unsigned c=0; c<PC2X2::COLS2X1; c++){
  for (unsigned s=0; s<PC2X2::N2X1_IN_DET; s++){

    int ix = int (pix_coords_2x2 -> getPixCoor_pix(PC2X2::AXIS_X, s, r, c) + 0.1);
    int iy = int (pix_coords_2x2 -> getPixCoor_pix(PC2X2::AXIS_Y, s, r, c) + 0.1);

    img_arr[ix][iy] = ix;
  }
  }
  }

  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(&img_arr[0][0],NY,NX);
  img2d -> saveImageInFile("test-img.txt",0);
}

//-----------------

void test03()
{
  const unsigned NX=400, NY=400;
  double img_arr[NY][NX];
  std::fill_n(&img_arr[0][0], int(NX*NY), double(0));

  //const std::string calibDir  = "/reg/d/psdm/mec/mec73313/calib";
  //const std::string groupName = "CsPad2x2::CalibV1";
  //const std::string source    = "MecTargetChamber.0:Cspad2x2.3";

  const std::string calibDir  = "/reg/d/psdm/xpp/xpptut13/calib";
  const std::string groupName = "CsPad2x2::CalibV1";
  const std::string source    = "XppGon.0:Cspad2x2.1";
  unsigned          runNumber = 10;
  CALIB2X2 *calibpars2x2 = new CALIB2X2(calibDir, groupName, source, runNumber);  
  calibpars2x2->printCalibPars();

  PC2X2 *pix_coords_2x2 = new PC2X2(calibpars2x2);  
  pix_coords_2x2 -> printXYLimits();

  // Assignment to coordinates

  for (unsigned r=0; r<PC2X2::ROWS2X1; r++){
  for (unsigned c=0; c<PC2X2::COLS2X1; c++){
  for (unsigned s=0; s<PC2X2::N2X1_IN_DET; s++){

    int ix = int (pix_coords_2x2 -> getPixCoor_pix(PC2X2::AXIS_X, s, r, c) + 0.1);
    int iy = int (pix_coords_2x2 -> getPixCoor_pix(PC2X2::AXIS_Y, s, r, c) + 0.1);

    img_arr[ix][iy] = ix+iy;
  }
  }
  }

  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(&img_arr[0][0],NY,NX);
  img2d -> saveImageInFile("test-img.txt",0);
}

//-----------------

int main ()
{
  //test01();
  //test02();
  test03();

  return 0;
}

//-----------------

