//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class PrincetonCalibPars of the PSCalib packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "PSCalib/CalibPars.h"
#include "PSCalib/GenericCalibPars.h"
#include "pdscalibdata/PrincetonBaseV1.h"

#include <string>
#include <iostream>

using namespace std;

int main ()
{
  // Assuming: /reg/d/psdm/xcs/xcstut13/calib/Princeton::CalibV1/XcsBeamline.0:Princeton.0/pedestals/1-end.data

  const std::string calibDir   = "/reg/d/psdm/xcs/xcstut13/calib";
  const std::string groupName  = "Princeton::CalibV1";
  const std::string source     = "XcsBeamline.0:Princeton.0";
  unsigned long     runNumber  = 20;
  unsigned          print_bits = 255; //0

  cout << "Test of PSCalib::PnccdCalibPars\n";     

  PSCalib::GenericCalibPars<pdscalibdata::PrincetonBaseV1> *calibpars = new PSCalib::GenericCalibPars<pdscalibdata::PrincetonBaseV1>(calibDir, groupName, source, runNumber, print_bits);  
  //PSCalib::GenericCalibPars<pdscalibdata::PrincetonBaseV1> *calibpars = new PSCalib::GenericCalibPars<pdscalibdata::PrincetonBaseV1>(true);  // test mode

  //calibpars->printCalibPars();
  //calibpars->printCalibParsStatus();
  calibpars->printInputPars();

  const PSCalib::CalibPars::pedestals_t* peds = calibpars->pedestals();
  for (unsigned i=0; i<10; i++) cout << peds[i] << "  "; cout << " ... - pedestals\n";

  const PSCalib::CalibPars::pixel_status_t* status = calibpars->pixel_status();
  for (unsigned i=0; i<10; i++) cout << status[i] << "  "; cout << " ... - pixel_status\n";

  const PSCalib::CalibPars::pixel_gain_t* gain = calibpars->pixel_gain();
  for (unsigned i=0; i<10; i++) cout << gain[i] << "  "; cout << " ... - pixel_gain\n";

  const PSCalib::CalibPars::pixel_rms_t* rms = calibpars->pixel_rms();
  for (unsigned i=0; i<10; i++) cout << rms[i] << "  "; cout << " ... - pixel_rms\n";

  const PSCalib::CalibPars::common_mode_t* cmod = calibpars->common_mode();
  for (unsigned i=0; i<pdscalibdata::PrincetonBaseV1::SizeCM; i++) cout << cmod[i] << "  "; cout << " - common_mode\n";

  calibpars -> printCalibParsStatus();
  calibpars -> printCalibPars();

  return 0;
}

//-----------------
