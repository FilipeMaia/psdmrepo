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

int main ()
{
  //const std::string calibDir   = "/reg/d/psdm/cxi/cxi35711/calib";
  //const std::string calibDir   = "/reg/d/psdm/cxi/cxi35711/calib_xxx"; // to test default pars
  const std::string calibDir   = "/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-test-calibpars";
  const std::string groupName  = "CsPad::CalibV1";
  const std::string source     = "CxiDs1.0:Cspad.0";
  unsigned long     runNumber  = 10;

  cout << "Test of PSCalib::CSPadCalibPars\n";     

  PSCalib::CSPadCalibPars *cspad_calibpars = new PSCalib::CSPadCalibPars(calibDir, groupName, source, runNumber);  

  cspad_calibpars->printCalibPars();

  return 0;
}

//-----------------
