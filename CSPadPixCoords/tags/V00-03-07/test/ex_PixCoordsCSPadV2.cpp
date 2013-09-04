//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class PixCoordsCSPadV2 of the CSPadPixCoords packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "PSCalib/CSPadCalibPars.h"
#include "CSPadPixCoords/CSPadConfigPars.h"
#include "CSPadPixCoords/PixCoordsCSPadV2.h"
#include "CSPadPixCoords/Image2D.h"

#include "ndarray/ndarray.h"

#include <string>
#include <iostream>
#include <time.h>

//using std::cout;
//using std::endl;

using namespace std;

typedef PSCalib::CSPadCalibPars CALIB;
typedef CSPadPixCoords::CSPadConfigPars CONFIG;
typedef CSPadPixCoords::PixCoordsCSPadV2 PC;

//-----------------

void test01()
{
  PC *pix_coords = new PC();  
  pix_coords -> printXYLimits();
}

//-----------------
// Makes image of pixel coordinates for defauld calibration and configuration parameters
void test02()
{
  PC *pix_coords = new PC();  
  pix_coords -> printXYLimits();

  // Reservation of memory for image array
  unsigned NX = (unsigned)(pix_coords -> get_x_max() * PC::UM_TO_PIX + 1); 
  unsigned NY = (unsigned)(pix_coords -> get_x_max() * PC::UM_TO_PIX + 1);   
  double* img_arr = new double[NX*NY];
  std::fill_n(img_arr, int(NX*NY), double(0));

  // Assignment to coordinates
  for (unsigned s=0; s<PC::N2X1_IN_DET; s++){
  for (unsigned r=0; r<PC::ROWS2X1; r++){
  for (unsigned c=0; c<PC::COLS2X1; c++){

    int ix = int (pix_coords -> getPixCoor_pix(PC::AXIS_X, s, r, c) + 0.1);
    int iy = int (pix_coords -> getPixCoor_pix(PC::AXIS_Y, s, r, c) + 0.1);

    //img_arr[ix][iy] = ix;
    img_arr[ix + iy*NX] = ix;
  }
  }
  }

  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(img_arr,NY,NX);
  img2d -> saveImageInFile("test-img.txt",0);
}

//-----------------
// Makes image of pixel coordinates for specified calibration and default configuration parameters

void test03()
{
  // /reg/d/psdm/cxi/cxitut13/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/
  const std::string calibDir  = "/reg/d/psdm/cxi/cxitut13/calib";
  const std::string groupName = "CsPad::CalibV1";
  const std::string source    = "CxiDs1.0:Cspad.0";
  unsigned          runNumber = 10;
  CALIB *calibpars = new CALIB(calibDir, groupName, source, runNumber);  
  calibpars->printCalibPars();

  PC *pix_coords = new PC(calibpars);  
  pix_coords -> printXYLimits();


  // Reservation of memory for image array
  unsigned NX = (unsigned)(pix_coords -> get_x_max() * PC::UM_TO_PIX + 1); 
  unsigned NY = (unsigned)(pix_coords -> get_x_max() * PC::UM_TO_PIX + 1);   
  double* img_arr = new double[NX*NY];
  std::fill_n(img_arr, int(NX*NY), double(0));

  // Assignment to coordinates
  for (unsigned s=0; s<PC::N2X1_IN_DET; s++){
  for (unsigned r=0; r<PC::ROWS2X1; r++){
  for (unsigned c=0; c<PC::COLS2X1; c++){

    int ix = int (pix_coords -> getPixCoor_pix(PC::AXIS_X, s, r, c) + 0.1);
    int iy = int (pix_coords -> getPixCoor_pix(PC::AXIS_Y, s, r, c) + 0.1);

    img_arr[ix + iy*NX] = r+c;
  }
  }
  }

  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(img_arr,NY,NX);
  img2d -> saveImageInFile("test-img.txt",0);
}

//-----------------
// Makes image of pixel coordinates for specified calibration and configuration parameters

void test04()
{
  // /reg/d/psdm/cxi/cxitut13/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/
  const std::string calibDir  = "/reg/d/psdm/cxi/cxitut13/calib";
  const std::string groupName = "CsPad::CalibV1";
  const std::string source    = "CxiDs1.0:Cspad.0";
  unsigned          runNumber = 10;
  CALIB *calibpars = new CALIB(calibDir, groupName, source, runNumber);  
  calibpars->printCalibPars();

  PC *pix_coords = new PC(calibpars);  
  pix_coords -> printXYLimits();

  uint32_t numQuads         = 4;                     // 4; 
  uint32_t quadNumber[]     = {0,1,2,3};             // {0,1,2,3};
  uint32_t roiMask[]        = {0375,0337,0177,0376}; // {0377,0377,0377,0377};
  CONFIG *config = new CONFIG( numQuads, quadNumber, roiMask );  
  config -> printCSPadConfigPars();


  // Reservation of memory for image array
  unsigned NX = (unsigned)(pix_coords -> get_x_max() * PC::UM_TO_PIX + 1); 
  unsigned NY = (unsigned)(pix_coords -> get_x_max() * PC::UM_TO_PIX + 1);   
  double* img_arr = new double[NX*NY];
  std::fill_n(img_arr, int(NX*NY), double(0));


  ndarray<double,3> nda_pix_coord_x = pix_coords -> getPixCoorNDArrShapedAsData_um (PC::AXIS_X, config);
  ndarray<double,3> nda_pix_coord_y = pix_coords -> getPixCoorNDArrShapedAsData_um (PC::AXIS_Y, config);

  // Assignment to coordinates using 3 indexes
  /*
  const unsigned* shape = nda_pix_coord_x.shape();

  for (unsigned s=0; s<shape[0]; s++) {
  for (unsigned r=0; r<PC::ROWS2X1; r++) {
  for (unsigned c=0; c<PC::COLS2X1; c++) {

    int ix = int ( nda_pix_coord_x[s][r][c] * PC::UM_TO_PIX + 0.1);
    int iy = int ( nda_pix_coord_y[s][r][c] * PC::UM_TO_PIX + 0.1);

    img_arr[ix + iy*NX] = r+c;
  }
  }
  }
  */

  // Assignment to coordinates for entire array
  int ix, iy;
  ndarray<double, 3>::iterator xit;
  ndarray<double, 3>::iterator yit;
  for(xit=nda_pix_coord_x.begin(), yit=nda_pix_coord_y.begin(); xit!=nda_pix_coord_x.end(); ++xit, ++yit) { 
    ix = int ( *xit * PC::UM_TO_PIX + 0.1);
    iy = int ( *yit * PC::UM_TO_PIX + 0.1);
    img_arr[ix + iy*NX] = ix+iy;
  }


  CSPadPixCoords::Image2D<double> *img2d = new CSPadPixCoords::Image2D<double>(img_arr,NY,NX);
  img2d -> saveImageInFile("test-img.txt",0);
}

//-----------------

int main (int argc, char* argv[])
{  
  cout << "Number of input arguments = " << argc << endl; 

  if      (argc == 1)                     {test03();}
  else if (argc == 2 && atoi(argv[1])==1) {test01();}
  else if (argc == 2 && atoi(argv[1])==2) {test02();}
  else if (argc == 2 && atoi(argv[1])==3) {test03();}
  else if (argc == 2 && atoi(argv[1])==4) {test04();}
  else    {
        cout << "WARNING!!! Unexpected input arguments, argc=" << argc << endl; 
        for(int i = 0; i < argc; i++)
            cout << "argv[" << i << "] = " << argv[i] << endl;
        cout << "Use command: " << argv[0] << " N, where N stands for test number 1,2,3,...\n"; 
  }
  return 0;
}

//-----------------

