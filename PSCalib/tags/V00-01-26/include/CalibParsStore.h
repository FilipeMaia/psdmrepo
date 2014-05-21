#ifndef PSCALIB_CALIBPARSSTORE_H
#define PSCALIB_CALIBPARSSTORE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// $Revision$
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <string>
//#include <vector>
//#include <map>
//#include <fstream>  // open, close etc.

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
#include "pdsdata/xtc/Src.hh"
#include "MsgLogger/MsgLogger.h"
#include "ImgAlgos/GlobalMethods.h" // ::toString( const Pds::Src& src )
#include "PSCalib/CalibPars.h"

#include "PSCalib/PnccdCalibPars.h"
#include "PSCalib/PrincetonCalibPars.h"
#include "PSCalib/CSPad2x2CalibIntensity.h"
#include "PSCalib/CSPadCalibIntensity.h"

//-----------------------------

namespace PSCalib {

/**
 *  @defgroup PSCalib PSCalib package
 *  @brief Package PSCalib provides access to the calibration parameters of all detectors
 */

/// @addtogroup PSCalib PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief class CalibParsStore has a static factory method Create for CalibPars
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @see CalibPars
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Includes
 *  @code
 *  #include "psana/Module.h" // for evt, env, get,  etc.
 *  #include "PSCalib/CalibPars.h"
 *  #include "PSCalib/CalibParsStore.h"
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  Here we assume that code is working inside psana module where evt and env variables are defined through input parameters of call-back methods. 
 *  Code below instateates calibpars object using factory static method PSCalib::CalibParsStore::Create:
 *  @code
 *  std::string calib_dir = env.calibDir(); // or "/reg/d/psdm/<INS>/<experiment>/calib"
 *  std::string  group = std::string(); // or something like "PNCCD::CalibV1";
 *  const std::string source = "Camp.0:pnCCD.1";
 *  const std::string key = ""; // key for raw data
 *  Pds::Src src; env.get(source, key, &src);
 *  PSCalib::CalibPars* calibpars = PSCalib::CalibParsStore::Create(calib_dir, group, src, PSCalib::getRunNumber(evt));
 *  @endcode
 *
 *  @li Access methods
 *  @code
 *  calibpars->printCalibPars();
 *  const PSCalib::CalibPars::pedestals_t*    peds_data = calibpars->pedestals();
 *  const PSCalib::CalibPars::pixel_gain_t*   gain_data = calibpars->pixel_gain();
 *  const PSCalib::CalibPars::pixel_rms_t*    rms_data  = calibpars->pixel_rms();
 *  const PSCalib::CalibPars::pixel_status_t* mask_data = calibpars->pixel_status();
 *  const PSCalib::CalibPars::common_mode_t*  cmod_data = calibpars->common_mode();
 *  @endcode
 */

//----------------

class CalibParsStore  {
public:

  //CalibParsStore () {}
  //virtual ~CalibParsStore () {}

  /**
   *  @brief Regular constructor, which use Pds::Src& src
   *  
   *  @param[in] calibdir       Calibration directory for current experiment.
   *  @param[in] group          Data type and group names.
   *  @param[in] src            The data source object, for example Pds::Src m_src; defined in the env.get(...,&m_src)
   *  @param[in] runnum         Run number to search the valid file name.
   *  @param[in] print_bits     Print control bit-word.
   */ 
  static PSCalib::CalibPars*
  Create ( const std::string&   calibdir,     //  /reg/d/psdm/mec/mec73313/calib
           const std::string&   group,        //  CsPad2x2::CalibV1
           const Pds::Src&      src,          //  Pds::Src m_src; <- is defined in env.get(...,&m_src)
           const unsigned long& runnum,       //  10
           unsigned             print_bits=255 )
  {

        std::string str_src = ImgAlgos::srcToString(src); 

        if ( str_src.find(":Cspad.") != std::string::npos ) {
           MsgLog("CalibParsStore", info, "Get access to calibration store for Cspad source: " << str_src);
	   std::string type_group = (group==std::string()) ? "CsPad::CalibV1" : group;
	   unsigned prbits = (print_bits & 1) ? 255 : 0;
	   return new PSCalib::CSPadCalibIntensity(calibdir, type_group, src, runnum, prbits);
	}

        if ( str_src.find(":Cspad2x2.") != std::string::npos ) {
           MsgLog("CalibParsStore", info, "Get access to calibration store for Cspad2x2 source: " << str_src);
	   std::string type_group = (group==std::string()) ? "CsPad2x2::CalibV1" : group;
	   unsigned prbits = (print_bits & 2) ? 255 : 0;
	   return new PSCalib::CSPad2x2CalibIntensity(calibdir, type_group, src, runnum, prbits);
	}

        if ( str_src.find(":pnCCD.") != std::string::npos ) {
           MsgLog("CalibParsStore", info, "Get access to calibration store for pnCCD source: " << str_src);
	   std::string type_group = (group==std::string()) ? "PNCCD::CalibV1" : group;
	   unsigned prbits = (print_bits & 4) ? 255 : 0;
	   return new PSCalib::PnccdCalibPars(calibdir, type_group, src, runnum, prbits);
	}

        if ( str_src.find(":Princeton.") != std::string::npos ) {
           MsgLog("CalibParsStore", info, "Get access to calibration store for Princeton source: " << str_src);
	   std::string type_group = (group==std::string()) ? "Princeton::CalibV1" : group;
	   unsigned prbits = (print_bits & 8) ? 40 : 0;
	   return new PSCalib::PrincetonCalibPars(calibdir, type_group, src, runnum, prbits);
	}

	// "CsPad::CalibV1"
	// "CsPad2x2::CalibV1"
	// "Princeton::CalibV1'"
	// "PNCCD::CalibV1"
	// "Camera::CalibV1"
	// "Acqiris::CalibV1"

	std::string msg =  "Calibration parameters for source: " + str_src + " are not implemented yet...";
        MsgLog("CalibParsStore", error, msg);  

        abort();

        //return NULL;
  }
};

} // namespace PSCalib

#endif // PSCALIB_CALIBPARSSTORE_H
