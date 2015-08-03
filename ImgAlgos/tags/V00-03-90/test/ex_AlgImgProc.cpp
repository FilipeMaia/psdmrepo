//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class ImgAlgos/AlgImgProc
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "ImgAlgos/AlgImgProc.h"

#include <string>
#include <iostream>

using std::cout;
using std::endl;

//-----------------

int test01 ()
{
  cout << "Test of ImgAlgos::AlgImgProc\n";     

  size_t      seg      = 2;
  size_t      rowmin   = 10;
  size_t      rowmax   = 170;
  size_t      colmin   = 100;
  size_t      colmax   = 200;
  unsigned    pbits    = 0; // 0177777;

  float       r0       = 5;
  float       dr       = 0.05;
 
  ImgAlgos::AlgImgProc* aip = new ImgAlgos::AlgImgProc(seg, rowmin, rowmax, colmin, colmax, pbits);

  aip->setSoNPars(r0,dr);

  aip->printInputPars();
  aip->printMatrixOfRingIndexes();
  aip->printVectorOfRingIndexes();

  return 0;
}

//-----------------

int main (int argc, char* argv[])
{  
  cout << "Number of input arguments = " << argc << endl; 
  // atoi(argv[1])==1) 
  
  test01();

  return 0;
}

//-----------------
