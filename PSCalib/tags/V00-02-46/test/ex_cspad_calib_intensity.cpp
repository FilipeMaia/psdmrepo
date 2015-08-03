//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class CSPadCalibIntensity of the PSCalib packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "PSCalib/CSPadCalibIntensity.h"

#include <string>
#include <iostream>

using std::cout;
using std::endl;

int main ()
{
  const std::string calibDir   = "/reg/d/psdm/xpp/xpptut13/calib";
  const std::string groupName  = "CsPad::CalibV1";
  const std::string source     = "XppGon.0:Cspad.0";
  unsigned long     runNumber  = 10;
  unsigned          print_bits = 255; //0

  cout << "Test of PSCalib::CSPadCalibIntensity\n";     

  PSCalib::CSPadCalibIntensity *calibpars = new PSCalib::CSPadCalibIntensity(calibDir, groupName, source, runNumber, print_bits);  
  //PSCalib::CSPadCalibIntensity *calibpars = new PSCalib::CSPadCalibIntensity(true);  // test mode

  calibpars->printCalibPars();
  calibpars->printCalibParsStatus();
  calibpars->printInputPars();

  return 0;
}

//-----------------
