#ifndef PSCALIB_CALIBPARSSTORE_H
#define PSCALIB_CALIBPARSSTORE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibParsStore.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <string>
//#include <vector>
//#include <map>
//#include <fstream>  // open, close etc.

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
#include "pdsdata/xtc/Src.hh"
#include "MsgLogger/MsgLogger.h"
#include "ImgAlgos/GlobalMethods.h" // ::toString( const Pds::Src& src )
#include "PSCalib/CalibPars.h"

#include "PSCalib/PnccdCalibPars.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSCalib {

/// @addtogroup PSCalib PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief class CalibParsStore has a static factory method Create for CalibPars
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CalibFileFinder
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
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
   */ 
  static PSCalib::CalibPars*
  Create ( const std::string&   calibdir,     //  /reg/d/psdm/mec/mec73313/calib
           const std::string&   group,        //  CsPad2x2::CalibV1
           const Pds::Src&      src,          //  Pds::Src m_src; <- is defined in env.get(...,&m_src)
           const unsigned long& runnum,       //  10
           unsigned             print_bits=255 )
  {

        std::string str_src = ImgAlgos::srcToString(src); 
        MsgLog("CalibParsStore", info, "Get calibration parameters for source: " << str_src);  

        if ( str_src.find(":pnCCD.") != std::string::npos ) {
           MsgLog("CalibParsStore", info, "Load calibration parameters for pnCCD");
	   std::string type_group = (group==std::string()) ? "PNCCD::CalibV1" : group;
	   return new PSCalib::PnccdCalibPars(calibdir, type_group, src, runnum, print_bits);
	}

	std::string msg =  "Calibration parameters for source: " + str_src + " are not implemented yet...";
        MsgLog("CalibParsStore", error, msg);  

        abort();

        //return NULL;
  }
};

} // namespace PSCalib

#endif // PSCALIB_CALIBPARSSTORE_H
