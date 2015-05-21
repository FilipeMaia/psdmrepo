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

#include "PSCalib/CSPadCalibPars.h"

#include <string>
#include <iostream>

using std::cout;
using std::endl;

//using namespace PSTime;
typedef PSCalib::CSPadCalibPars CALIB;

//-----------------

int test01 () // Test for XPP
{
  const std::string calibDir   = "/reg/d/psdm/xpp/xpptut13/calib";
  const std::string groupName  = "CsPad::CalibV1";
  const std::string source     = "XppGon.0:Cspad.0";
  unsigned long     runNumber  = 10;

  cout << "Test of PSCalib::CSPadCalibPars\n";     

  CALIB *calibpars = new CALIB(calibDir, groupName, source, runNumber);  

  calibpars->printCalibPars();
  calibpars->printCalibParsStatus();

  size_t quad=1, sect=7; // for example...
  cout << "\ngetCenterGlobal[pix] =" << calibpars -> getCenterGlobalX(quad, sect) << " for quad=" << quad << " sect=" << sect << "\n";  

  return 0;
}

//-----------------

int test02 () // Test for CXI
{
  //const std::string calibDir   = "/reg/d/psdm/cxi/cxi35711/calib";
  //const std::string calibDir   = "/reg/d/psdm/cxi/cxi35711/calib_xxx"; // to test default pars
  //const std::string calibDir   = "/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-test-calibpars";
  const std::string calibDir   = "/reg/d/psdm/cxi/cxitut13/calib";
  const std::string groupName  = "CsPad::CalibV1";
  const std::string source     = "CxiDs1.0:Cspad.0";
  unsigned long     runNumber  = 10;

  cout << "Test of PSCalib::CSPadCalibPars\n";     

  CALIB *calibpars = new CALIB(calibDir, groupName, source, runNumber);  

  calibpars->printCalibPars();
  calibpars->printCalibParsStatus();

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
