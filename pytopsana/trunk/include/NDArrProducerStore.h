#ifndef PYTOPSANA_NDARRPRODUCERSTORE_H
#define PYTOPSANA_NDARRPRODUCERSTORE_H

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
#include "MsgLogger/MsgLogger.h"

#include "pytopsana/NDArrProducerBase.h"
#include "pytopsana/NDArrProducerCSPAD.h"
//#include "pytopsana/NDArrProducerPNCCD.h"


//#include "pdsdata/xtc/Src.hh"
//#include "PSCalib/CalibPars.h"

//#include "PSCalib/CSPad2x2CalibIntensity.h"
//#include "PSCalib/CSPadCalibIntensity.h"
//#include "PSCalib/PnccdCalibPars.h"
//#include "PSCalib/PrincetonCalibPars.h"


//#include "ImgAlgos/GlobalMethods.h" // for DETECTOR_TYPE, getRunNumber(evt), detectorTypeForSource, etc.
//#include "PSCalib/GenericCalibPars.h"

//#include "pdscalibdata/CsPadBaseV2.h"     // shape_base(), Ndim, Rows, Cols, Size, etc.
//#include "pdscalibdata/CsPad2x2BaseV2.h"
//#include "pdscalibdata/PnccdBaseV1.h"
//#include "pdscalibdata/PrincetonBaseV1.h"
//#include "pdscalibdata/AndorBaseV1.h"
//#include "pdscalibdata/Epix100aBaseV1.h"
//#include "pdscalibdata/VarShapeCameraBaseV1.h"
//#include "pdscalibdata/Opal1000BaseV1.h"
//#include "pdscalibdata/Opal4000BaseV1.h"


//-----------------------------

namespace pytopsana {

/**
 *  @defgroup PSCalib PSCalib package
 *  @brief Package PSCalib provides access to the calibration parameters of all detectors
 */

/// @addtogroup PSCalib PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief class NDArrProducerStore has a static factory method Create for CalibPars
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
 *  #include "PSCalib/NDArrProducerStore.h"
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  Here we assume that code is working inside psana module where evt and env variables are defined through input parameters of call-back methods. 
 *  Code below instateates calibpars object using factory static method PSCalib::NDArrProducerStore::Create:
 *  @code
 *  std::string calib_dir = env.calibDir(); // or "/reg/d/psdm/<INS>/<experiment>/calib"
 *  std::string  group = std::string(); // or something like "PNCCD::CalibV1";
 *  const std::string source = "Camp.0:pnCCD.1";
 *  const std::string key = ""; // key for raw data
 *  Pds::Src src; env.get(source, key, &src);
 *  PSCalib::CalibPars* calibpars = PSCalib::NDArrProducerStore::Create(calib_dir, group, src, PSCalib::getRunNumber(evt));
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

class NDArrProducerStore  {


private:
  inline const char* name(){return "NDArrProducerStore";}

public:

  //NDArrProducerStore () {}
  //virtual ~NDArrProducerStore () {}
  /**
   *  @brief Regular constructor, which use const std::string& str_src
   *  
   *  @param[in] calibdir       Calibration directory for current experiment.
   *  @param[in] group          Data type and group names.
   *  @param[in] src            The data source name, ex.: Camp.0:pnCCD.0
   *  @param[in] runnum         Run number to search the valid file name.
   *  @param[in] print_bits     Print control bit-word.
   */ 
  static pytopsana::NDArrProducerBase*
  Create ( const PSEvt::Source& source,      //  Camp.0:pnCCD.0
           const unsigned& pbits=255 )
  {
        //unsigned prbits = (pbits & 8) ? 40 : 0;

        ImgAlgos::DETECTOR_TYPE m_dettype = ImgAlgos::detectorTypeForSource(source);  // numerated detector type defined from source string info
        if (pbits & 1) MsgLog("NDArrProducerStore", info, "Get access to CSPAD data source: " << source);


	if (m_dettype == ImgAlgos::CSPAD) {
	  return new NDArrProducerCSPAD(source);
	}


	/*
        if ( str_src.find(":Cspad.") != std::string::npos ) {
           MsgLog("NDArrProducerStore", info, "Get access to calibration store for Cspad source: " << str_src);
	   std::string type_group = (group==std::string()) ? "CsPad::CalibV1" : group;
	   unsigned prbits = (print_bits & 1) ? 255 : 0;
	   return new PSCalib::CSPadCalibIntensity(calibdir, type_group, src, runnum, prbits);
	}
	*/

	// Generic approach to calibration 

	/*
        if ( str_src.find(":Cspad.") != std::string::npos ) {
           if (print_bits & 1) MsgLog("NDArrProducerStore", info, "Get access to calibration store for Cspad source: " << str_src);
	   std::string type_group = (group==std::string()) ? "CsPad::CalibV1" : group;
	   return new PSCalib::GenericCalibPars<pdscalibdata::CsPadBaseV2>(calibdir, type_group, str_src, runnum, prbits);
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
            if (print_bits & 1) MsgLog("NDArrProducerStore", info, "Get access to calibration store for det " << *it 
                                           << " source: " << str_src << " group: " << type_group);
	    return new PSCalib::GenericCalibPars<pdscalibdata::VarShapeCameraBaseV1>(calibdir, type_group, str_src, runnum, prbits);
	    //return new PSCalib::GenericCalibPars<pdscalibdata::Opal1000BaseV1>(calibdir, type_group, src, runnum, prbits);
	  }
	}
	*/

        MsgLog("NDArrProducerStore", error, "Access to data for source " << source << " is not implemented yet...");  
        abort();

        return NULL;
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

  /*
  static PSCalib::CalibPars*
  Create ( const std::string&   calibdir,     //  /reg/d/psdm/mec/mec73313/calib
           const std::string&   group,        //  CsPad2x2::CalibV1
           const Pds::Src&      src,          //  Pds::Src m_src; <- is defined in env.get(...,&m_src)
           const unsigned long& runnum,       //  10
           unsigned             print_bits=255 )
  {
    return Create(calibdir, group, ImgAlgos::srcToString(src), runnum, print_bits);
  }
  */

}; // class

} // namespace

#endif // PYTOPSANA_NDARRPRODUCERSTORE
