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
#include <vector>
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

//#include "PSCalib/CSPad2x2CalibIntensity.h"
//#include "PSCalib/CSPadCalibIntensity.h"
//#include "PSCalib/PnccdCalibPars.h"
//#include "PSCalib/PrincetonCalibPars.h"


#include "PSCalib/GenericCalibPars.h"

#include "pdscalibdata/CsPadBaseV2.h"     // shape_base(), Ndim, Rows, Cols, Size, etc.
#include "pdscalibdata/CsPad2x2BaseV2.h"
#include "pdscalibdata/PnccdBaseV1.h"
#include "pdscalibdata/PrincetonBaseV1.h"
#include "pdscalibdata/AndorBaseV1.h"
#include "pdscalibdata/Epix100aBaseV1.h"
#include "pdscalibdata/VarShapeCameraBaseV1.h"
//#include "pdscalibdata/Opal1000BaseV1.h"
//#include "pdscalibdata/Opal4000BaseV1.h"


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
 *  const PSCalib::CalibPars::pixel_mask_t*   mask_data = calibpars->pixel_mask();
 *  const PSCalib::CalibPars::pixel_bkgd_t*   bkgd_data = calibpars->pixel_bkgd();
 *  const PSCalib::CalibPars::pixel_rms_t*    rms_data  = calibpars->pixel_rms();
 *  const PSCalib::CalibPars::pixel_status_t* stat_data = calibpars->pixel_status();
 *  const PSCalib::CalibPars::common_mode_t*  cmod_data = calibpars->common_mode();
 *  @endcode
 */

//----------------

class CalibParsStore  {
public:

  //CalibParsStore () {}
  //virtual ~CalibParsStore () {}


  /**
   *  @brief Regular constructor, which use const std::string& str_src
   *  
   *  @param[in] calibdir       Calibration directory for current experiment.
   *  @param[in] group          Data type and group names.
   *  @param[in] str_src        The data source name, ex.: Camp.0:pnCCD.0
   *  @param[in] runnum         Run number to search the valid file name.
   *  @param[in] print_bits     Print control bit-word.
   */ 
  static PSCalib::CalibPars*
  Create ( const std::string&   calibdir,     //  /reg/d/psdm/AMO/amoa1214/calib
           const std::string&   group,        //  PNCCD::CalibV1
           const std::string&   str_src,      //  Camp.0:pnCCD.0
           const unsigned long& runnum,       //  10
           unsigned             print_bits=255 )
  {
	unsigned prbits = (print_bits & 8) ? 255 : 0;

	/*
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
	*/

	// Generic approach to calibration 

        if ( str_src.find(":Cspad.") != std::string::npos ) {
           if (print_bits & 1) MsgLog("CalibParsStore", info, "Get access to calibration store for Cspad source: " << str_src);
	   std::string type_group = (group==std::string()) ? "CsPad::CalibV1" : group;
	   return new PSCalib::GenericCalibPars<pdscalibdata::CsPadBaseV2>(calibdir, type_group, str_src, runnum, prbits);
	}

        if ( str_src.find(":Cspad2x2.") != std::string::npos ) {
           if (print_bits & 1) MsgLog("CalibParsStore", info, "Get access to calibration store for Cspad2x2 source: " << str_src);
	   std::string type_group = (group==std::string()) ? "CsPad2x2::CalibV1" : group;
	   return new PSCalib::GenericCalibPars<pdscalibdata::CsPad2x2BaseV2>(calibdir, type_group, str_src, runnum, prbits);
	}

        if ( str_src.find(":pnCCD.") != std::string::npos ) {
           if (print_bits & 1) MsgLog("CalibParsStore", info, "Get access to calibration store for pnCCD source: " << str_src);
	   std::string type_group = (group==std::string()) ? "PNCCD::CalibV1" : group;
	   return new PSCalib::GenericCalibPars<pdscalibdata::PnccdBaseV1>(calibdir, type_group, str_src, runnum, prbits);
	}

        if ( str_src.find(":Princeton.") != std::string::npos ) {
           if (print_bits & 1) MsgLog("CalibParsStore", info, "Get access to calibration store for Princeton source: " << str_src);
	   std::string type_group = (group==std::string()) ? "Princeton::CalibV1" : group;
	   return new PSCalib::GenericCalibPars<pdscalibdata::PrincetonBaseV1>(calibdir, type_group, str_src, runnum, prbits);
	}

        if ( str_src.find(":Andor.") != std::string::npos ) {
           if (print_bits & 1) MsgLog("CalibParsStore", info, "Get access to calibration store for Andor source: " << str_src);
	   std::string type_group = (group==std::string()) ? "Andor::CalibV1" : group;
	   return new PSCalib::GenericCalibPars<pdscalibdata::AndorBaseV1>(calibdir, type_group, str_src, runnum, prbits);
	}

        if ( str_src.find(":Epix100a.") != std::string::npos ) {
           if (print_bits & 1) MsgLog("CalibParsStore", info, "Get access to calibration store for Epix100a source: " << str_src);
	   std::string type_group = (group==std::string()) ? "Epix100a::CalibV1" : group;
	   return new PSCalib::GenericCalibPars<pdscalibdata::Epix100aBaseV1>(calibdir, type_group, str_src, runnum, prbits);
	}

	std::vector<std::string> v_camera_names;
	v_camera_names.push_back(":Opal1000.");
	v_camera_names.push_back(":Opal2000.");
	v_camera_names.push_back(":Opal4000.");
	v_camera_names.push_back(":Opal8000.");
	v_camera_names.push_back(":Tm6740.");
	v_camera_names.push_back(":OrcaFl40.");
	v_camera_names.push_back(":Fccd960.");

        prbits = (print_bits & 8) ? 255 : 0;

	for (std::vector<std::string>::iterator it = v_camera_names.begin(); it != v_camera_names.end(); ++it) {
          if ( str_src.find(*it) != std::string::npos ) {
	    std::string type_group = (group==std::string()) ? "Camera::CalibV1" : group;
            if (print_bits & 1) MsgLog("CalibParsStore", info, "Get access to calibration store for det " << *it 
                                           << " source: " << str_src << " group: " << type_group);
	    return new PSCalib::GenericCalibPars<pdscalibdata::VarShapeCameraBaseV1>(calibdir, type_group, str_src, runnum, prbits);
	    //return new PSCalib::GenericCalibPars<pdscalibdata::Opal1000BaseV1>(calibdir, type_group, src, runnum, prbits);
	  }
	}

	std::string msg = "Calibration parameters for source: " + str_src + " are not implemented yet...";
        MsgLog("CalibParsStore", error, msg);  

        abort();

        //return NULL;
  }



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
    return Create(calibdir, group, ImgAlgos::srcToString(src), runnum, print_bits);
  }

};

} // namespace PSCalib

#endif // PSCALIB_CALIBPARSSTORE_H
