//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class CSPad2x2CalibIntensity of the PSCalib packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "PSCalib/CSPad2x2CalibIntensity.h"

#include <string>
#include <iostream>

using std::cout;
using std::endl;

int main ()
{
  const std::string calibDir   = "/reg/d/psdm/xpp/xpptut13/calib";
  const std::string groupName  = "CsPad2x2::CalibV1";
  const std::string source     = "XppGon.0:Cspad2x2.0";
  unsigned long     runNumber  = 10;
  unsigned          print_bits = 255; //0

  cout << "Test of PSCalib::CSPad2x2CalibIntensity\n";     

  PSCalib::CSPad2x2CalibIntensity *calibpars = new PSCalib::CSPad2x2CalibIntensity(calibDir, groupName, source, runNumber, print_bits);  
  //PSCalib::CSPad2x2CalibIntensity *calibpars = new PSCalib::CSPad2x2CalibIntensity(true);  // test mode

  calibpars->printCalibPars();
  calibpars->printCalibParsStatus();
  calibpars->printInputPars();

  return 0;
}

//-----------------
