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

#include "PSCalib/CSPad2x2CalibPars.h"

#include <string>
#include <iostream>

using std::cout;
using std::endl;

//using namespace PSTime;

int main ()
{
  // Assuming path: /reg/d/psdm/mec/mec73313/calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/1-end.data

  //const std::string calibDir   = "/reg/d/psdm/mec/mec73313/calib_xxx"; // to test default pars
  //const std::string calibDir   = "/reg/neh/home1/dubrovin/LCLS/CSPad2x2Alignment/calib-test-calibpars";
  const std::string calibDir   = "/reg/d/psdm/mec/mec73313/calib";
  const std::string groupName  = "CsPad2x2::CalibV1";
  const std::string source     = "MecTargetChamber.0:Cspad2x2.1";
  unsigned long     runNumber  = 10;

  cout << "Test of PSCalib::CSPad2x2CalibPars\n";     

  PSCalib::CSPad2x2CalibPars *cspad_calibpars = new PSCalib::CSPad2x2CalibPars(calibDir, groupName, source, runNumber);  

  cspad_calibpars->printCalibPars();

  return 0;
}

//-----------------
