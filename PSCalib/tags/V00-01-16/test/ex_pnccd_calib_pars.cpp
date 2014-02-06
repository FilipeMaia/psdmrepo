//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class PnccdCalibPars of the PSCalib packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "PSCalib/PnccdCalibPars.h"

#include <string>
#include <iostream>

using std::cout;
using std::endl;

//using namespace PSTime;

int main ()
{
  // Assuming path: /reg/d/psdm/AMO/amoa1214/calib/PNCCD::CalibV1/Camp.0:pnCCD.1/pedestals/1-end.data
  // or:            /reg/d/psdm/AMO/amotut13/calib/PNCCD::CalibV1/Camp.0:pnCCD.1/pedestals/1-end.data

  //const std::string calibDir   = "/reg/neh/home1/dubrovin/LCLS/.../calib-test-calibpars";
  //const std::string calibDir   = "/reg/d/psdm/AMO/amoa1214/calib";
  //const std::string calibDir   = "/reg/d/psdm/mec/mec73313/calib_xxx"; // to test default pars
  //const std::string calibDir   = "/reg/d/psdm/AMO/amoa1214/calib";
  const std::string calibDir   = "/reg/d/psdm/AMO/amotut13/calib";
  const std::string groupName  = "PNCCD::CalibV1";
  const std::string source     = "Camp.0:pnCCD.1";
  unsigned long     runNumber  = 10;
  unsigned          print_bits = 255; //0

  cout << "Test of PSCalib::PnccdCalibPars\n";     

  PSCalib::PnccdCalibPars *calibpars = new PSCalib::PnccdCalibPars(calibDir, groupName, source, runNumber, print_bits);  
  //PSCalib::PnccdCalibPars *calibpars = new PSCalib::PnccdCalibPars(true);  // test mode

  calibpars->printCalibPars();
  calibpars->printCalibParsStatus();
  calibpars->printInputPars();

  return 0;
}

//-----------------
