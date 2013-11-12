//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class CSPadConfigPars of the CSPadPixCoords packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "CSPadPixCoords/CSPadConfigPars.h"

//#include <string>
//#include <iostream>

//using std::cout;
//using std::endl;

using namespace std;

typedef CSPadPixCoords::CSPadConfigPars CONFIG;

//-----------------

void test01()
{
  CONFIG *config = new CONFIG();  

  config -> printCSPadConfigPars();

  cout << "  quad roi mask for quad 2 is " << config -> roiMask(2) << "\n"; 
}

//-----------------

void test02()
{
  uint32_t numQuads         = 3;                     // 4; 
  uint32_t quadNumber[]     = {0,1,3,3};             // {0,1,2,3};
  uint32_t roiMask[]        = {0377,0355,0377,0377}; // {0377,0377,0377,0377};
  CONFIG *config = new CONFIG( numQuads, quadNumber, roiMask );  

  config -> printCSPadConfigPars();

  cout << "  quad roi mask for quad 2 is " << config -> roiMask(2) << "\n"; 
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

