//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class CalibFileFinder of the PSCalib packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "PSCalib/CalibFileFinder.h"

#include <string>
#include <iostream>

using std::cout;
using std::endl;

//using namespace PSTime;

int main ()
{
  const std::string calibDir      = "/reg/d/psdm/cxi/cxi35711/calib";
  const std::string typeGroupName = "CsPad::CalibV1";
  const std::string src           = "CxiDs1.0:Cspad.0";
  const std::string dataType      = "pedestals";
  unsigned long     runNumber     = 10;
  unsigned long     runNumber2    = 50;

  cout << "Test of PSCalib::CalibFileFinder" << endl;     


  PSCalib::CalibFileFinder *calib1 = new PSCalib::CalibFileFinder(calibDir, typeGroupName);

  cout << "Calibration file name for existing run range: " << calib1->findCalibFile(src,dataType,runNumber) << endl;

  cout << "Calibration file name for non-existing run range: " << calib1->findCalibFile(src,dataType,runNumber2) << endl;

  return 0;
}

//-----------------
