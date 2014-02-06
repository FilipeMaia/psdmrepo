#ifndef PSCALIB_PNCCDCALIBPARS_H
#define PSCALIB_PNCCDCALIBPARS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnccdCalibPars.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <fstream>  // open, close etc.

//----------------------
// Base Class Headers --
//----------------------
#include "ndarray/ndarray.h"
#include "PSCalib/CalibPars.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/pnccd.ddl.h"
#include "pdsdata/xtc/Src.hh"

#include "pdscalibdata/PnccdBaseV1.h"      
#include "pdscalibdata/PnccdPedestalsV1.h"      
#include "pdscalibdata/PnccdCommonModeV1.h"        
#include "pdscalibdata/PnccdPixelStatusV1.h"        
#include "pdscalibdata/PnccdPixelGainV1.h"        

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
 *  @brief PnccdCalibPars class loads/holds/provides access to the CSPad2x2
 *  geometry calibration parameters.
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CalibFileFinder
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *
 *
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Include and typedef
 *  @code
 *  #include "PSCalib/PnccdCalibPars.h"
 *  typedef PSCalib::PnccdCalibPars CALIB;
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  For default constructor:
 *  @code
 *  CALIB *calibpars = new CALIB();  
 *  @endcode
 *  \n
 *  For regular constructor:
 *  @code
 *  const std::string calibDir   = "/reg/d/psdm/AMO/amoa1214/calib";
 *  const std::string groupName  = "PNCCD::CalibV1";
 *  const std::string source     = "Camp.0:pnCCD.1";
 *  unsigned long     runNumber  = 10;
 *  Pds::Src src; env.get(...,&src);
 *  CALIB *calibpars = new CALIB(calibDir, groupName, src, runNumber);  
 *  @endcode
 *  \n
 *  For explicit constructor (depricated):
 *  @code
 *  const std::string calibDir   = "/reg/d/psdm/AMO/amoa1214/calib";
 *  const std::string groupName  = "PNCCD::CalibV1";
 *  const std::string source     = "Camp.0:pnCCD.1";
 *  unsigned long     runNumber  = 10;
 *  CALIB *calibpars = new CALIB(calibDir, groupName, source, runNumber);  
 *  @endcode
 *
 *  @li Printing methods
 *  @code
 *  calibpars -> printInputPars();
 *  calibpars -> printCalibPars();
 *  calibpars -> printCalibParsStatus();
 *  @endcode
 *
 *  @li Access methods
 *  @code
 *  size_t sect=1; // for example...
 *  int status  = calibpars -> getCalibTypeStatus("pedestals") // Returns status: 0-default, 1-loaded from file
 *  ndarray<pdscalibdata::PnccdPedestalsV1::pars_t, 3>   nda_peds = calibpars -> pedestals();
 *  ndarray<pdscalibdata::PnccdPixelStatusV1::pars_t, 3> nda_stat = calibpars -> pixel_status();
 *  ndarray<pdscalibdata::PnccdCommonModeV1::pars_t, 1>  nda_cmod = calibpars -> common_mode();
 *  ndarray<pdscalibdata::PnccdPixelGainV1::pars_t, 3>   nda_gain = calibpars -> pixel_gain();
 *  ... etc. for all other access methods
 *  @endcode
 *
 *  @see CalibFileFinder
 */

//----------------

  class PnccdCalibPars: public PSCalib::CalibPars  {
public:

  /// Default and test constructor
  PnccdCalibPars ( bool isTestMode = false ) ;


  /**
   *  @brief DEPRICATED constructor, which use string& source
   *  
   *  @param[in] calibDir       Calibration directory for current experiment.
   *  @param[in] typeGroupName  Data type and group names.
   *  @param[in] source         The name of the data source.
   *  @param[in] runNumber      Run number to search the valid file name.
   *  @param[in] print_bits     =0-print no messages; +1-input parameters, +2-print msges from  PSCalib::CalibFileFinder, +4-use default, +8-missing type
   */ 
  PnccdCalibPars ( const std::string&   calibDir,           //  /reg/d/psdm/mec/mec73313/calib
                   const std::string&   typeGroupName,      //  CsPad2x2::CalibV1
                   const std::string&   source,             //  MecTargetChamber.0:Cspad2x2.1
                   const unsigned long& runNumber,          //  10
                   unsigned             print_bits=255 );

  /**
   *  @brief Regular constructor, which use Pds::Src& src
   *  
   *  @param[in] calibDir       Calibration directory for current experiment.
   *  @param[in] typeGroupName  Data type and group names.
   *  @param[in] src            The data source object, for example Pds::Src m_src; defined in the env.get(...,&m_src)
   *  @param[in] runNumber      Run number to search the valid file name.
   */ 
  PnccdCalibPars ( const std::string&   calibDir,           //  /reg/d/psdm/mec/mec73313/calib
                   const std::string&   typeGroupName,      //  CsPad2x2::CalibV1
                   const Pds::Src&      src,                //  Pds::Src m_src; <- is defined in env.get(...,&m_src)
                   const unsigned long& runNumber,          //  10
                   unsigned             print_bits=255 ) ;  

  /// Destructor
  virtual ~PnccdCalibPars () ;

  //size_t   getNRows             (){ return m_nrows;   };
  //size_t   getNCols             (){ return m_ncols;   };

  /// Makes member data vector with all supported calibration types such as center, tilt, ...
  void fillCalibNameVector   ();

  /// Define the path to the calibration file based on input parameters
  void getCalibFileName      ();

  /// Load all known calibration parameters
  void loadCalibPars         ();

  /// Fill calibration parameters from vector
  void fillCalibParsV1       ();

  /// Fill default calibration parameters
  void fillDefaultCalibParsV1();

  /// Generate error message in the log and abort
  void fatalMissingFileName  ();

  /// Generate warning message in the log
  void msgUseDefault         ();

  /// Prints calibration parameters
  void printCalibPars        (); // declared as pure virtual in superclass

  /// Prints input parameters of the object
  void printInputPars        ();

  /// Returns status of the calibration constants, 0-default, 1-loaded from file @param[in] type - calibration type string-name, for example "center" or "tilt"
  int getCalibTypeStatus(const std::string&  type) { return m_calibtype_status[type]; };



  /// INTERFACE METHODS

  /// Prints calibration parameters status
  virtual void printCalibParsStatus  ();

  /// Returns ndarray of pnCCD pedestals
  //pdscalibdata::PnccdPedestalsV1::pars_t
  //ndarray<CalibPars::pedestals_t, 3> pedestals(){ return m_pedestals -> pedestals(); };

  virtual const size_t    ndim() { return pdscalibdata::PnccdBaseV1::Ndim; };
  virtual const size_t    size() { return pdscalibdata::PnccdBaseV1::Size; };
  virtual const unsigned* shape(){ return m_pedestals -> pedestals().shape(); };

  virtual const CalibPars::pedestals_t* pedestals(){ return m_pedestals -> pedestals().data(); };

  /// Returns ndarray of pnCCD pixel status
  //pdscalibdata::PnccdPixelStatusV1::pars_t
  //ndarray<CalibPars::pixel_status_t, 3> pixel_status(){ return m_pixel_status -> pixel_status(); };
  virtual const CalibPars::pixel_status_t* pixel_status(){ return m_pixel_status -> pixel_status().data(); };

  /// Returns ndarray of pnCCD common mode
  //ndarray<CalibPars::common_mode_t, 1> common_mode(){ return m_common_mode -> common_mode(); };
  virtual const CalibPars::common_mode_t* common_mode(){ return m_common_mode -> common_mode().data(); };

  /// Returns ndarray of pnCCD pixel gain
  //ndarray<CalibPars::pixel_gain_t, 3> pixel_gain(){ return m_pixel_gain -> pixel_gain(); };
  virtual const CalibPars::pixel_gain_t* pixel_gain(){ return m_pixel_gain -> pixel_gain().data(); };


private:

  /// Copy constructor is disabled by default
  PnccdCalibPars ( const PnccdCalibPars& ) ;
  /// Assignment is disabled by default
  //PnccdCalibPars operator = ( const PnccdCalibPars& ) ;

//------------------
// Static Members --
//------------------

  // Assuming path: /reg/d/psdm/AMO/amoa1214/calib/PNCCD::CalibV1/Camp.0:pnCCD.1/pedestals/1-end.data

  std::string   m_calibdir;       // /reg/d/psdm/AMO/amoa1214/calib
  std::string   m_calibfilename;  // 1-end.data

  // Data members for regular constructor 
  std::string   m_calibDir;
  std::string   m_typeGroupName;
  std::string   m_source;
  Pds::Src      m_src;
  std::string   m_dataType;
  unsigned long m_runNumber;
  unsigned      m_print_bits;

  std::vector<std::string> v_calibname; // center, tilt, ...

  std::map<std::string, int> m_calibtype_status; // =0-default, =1-from file

  std::string m_cur_calibname;  
  std::string m_fname;

  bool m_isTestMode;

  //size_t m_nrows; 
  //size_t m_ncols; 

  std::ifstream m_file;

  pdscalibdata::PnccdPedestalsV1   *m_pedestals;
  pdscalibdata::PnccdPixelGainV1   *m_pixel_gain;
  pdscalibdata::PnccdCommonModeV1  *m_common_mode;
  pdscalibdata::PnccdPixelStatusV1 *m_pixel_status; 

};

} // namespace PSCalib

#endif // PSCALIB_PNCCDCALIBPARS_H
