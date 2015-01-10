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

#include "PSCalib/SegGeometryStore.h"

#include <string>
#include <iostream>
#include <stdlib.h> // for atoi

//using std::cout;
//using std::endl;
//using std::atoi;
using namespace std;

typedef PSCalib::SegGeometry SG;

//-----------------

int test01 () // Test 01 for CSPAD SENS2X1:V1
{
  cout << "Test of PSCalib::SegGeometryStore::Create(...)\n";     

  SG *seggeom = PSCalib::SegGeometryStore::Create("SENS2X1:V1", 0377);  
  seggeom -> print_seg_info(0377);

  return 0;
}

//-----------------

int test02 () // Test 02 for EPIX100:V1
{
  cout << "Test of PSCalib::SegGeometryStore::Create(...)\n";     

  SG *seggeom = PSCalib::SegGeometryStore::Create("EPIX100:V1", 0377);  
  seggeom -> print_seg_info(0377);

  return 0;
}

//-----------------

int main (int argc, char* argv[])
{  
  cout << "Number of input arguments = " << argc << endl; 

  if      (argc == 1)                     {test01();}
  else if (argc == 2 && atoi(argv[1])==1) {test01();}
  else if (argc == 2 && atoi(argv[1])==2) {test02();}
  else    {
        cout << "WARNING!!! Unexpected input arguments, argc=" << argc << endl; 
        for(int i = 0; i < argc; i++)
            cout << "argv[" << i << "] = " << argv[i] << endl;
        cout << "Use command: " << argv[0] << " N, where N stands for test number 1,2,3,...\n"; 
  }
  return 0;
}

//-----------------
