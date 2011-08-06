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
  const std::string calibDir      = "/reg/d/psdm/cxi/cxi35711/calib";
  const std::string typeGroupName = "CsPad::CalibV1";
  const std::string source        = "CxiDs1.0:Cspad.0";
  const std::string dataType      = "pedestals";
  unsigned long     runNumber     = 10;

  cout << "Test of PSCalib::CSPadCalibPars" << endl;     


  PSCalib::CSPadCalibPars *cspad_calibpars = new PSCalib::CSPadCalibPars(calibDir, typeGroupName, source, runNumber);  

  cspad_calibpars->printCalibPars();


  return 0;
}

//-----------------
