#ifndef PSCALIB_PRINCETONCALIBPARS_H
#define PSCALIB_PRINCETONCALIBPARS_H

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
#include <map>
#include <fstream>  // open, close etc.

//----------------------
// Base Class Headers --
//----------------------
#include "PSCalib/CalibPars.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
#include "pdsdata/xtc/Src.hh"
//#include "psddl_psana/princeton.ddl.h"

#include "pdscalibdata/PrincetonBaseV1.h" // shape_base(), Ndim, Rows, Cols, Size, etc.
#include "pdscalibdata/NDArrIOV1.h"      

//-------------------------------

namespace PSCalib {

/// @addtogroup PSCalib PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief PrincetonCalibPars class loads/holds/provides access to the pnCCD
 *  geometry calibration parameters.
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CalibPars, CalibParsStore
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
 *  #include "PSCalib/PrincetonCalibPars.h"
 *  typedef PSCalib::PrincetonCalibPars CALIB;
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
 *  Pds::Src src; env.get(source, key, &src);
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
 *  \n
 *  or for the same list of parameters using polymorphism:
 *  @code
 *  PSCalib::CalibPars *calibpars = new CALIB(calibDir, groupName, source, runNumber);  
 *  @endcode
 *  In this case all virtual methods defined in the base class PSCalib::CalibPars will be accessible through the calibpars pointer.
 *  
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
 *  const size_t    ndim = calibpars -> ndim();
 *  const size_t    size = calibpars -> size();
 *  const unsigned* p_shape = calibpars -> shape();
 *  const CalibPars::pedestals_t*    p_pedestals   = calibpars -> pedestals()
 *  const CalibPars::pixel_status_t* p_pixel_stat  = calibpars -> pixel_status()
 *  const CalibPars::common_mode_t*  p_common_mode = calibpars -> common_mode()
 *  const CalibPars::pixel_gain_t*   p_pixel_gain  = calibpars -> pixel_gain()
 *  const CalibPars::pixel_rms_t*    p_pixel_rms   = calibpars -> pixel_rms()
 *  ... etc. for all other access methods
 *  @endcode
 *
 *  @see CalibFileFinder
 */

//----------------

  class PrincetonCalibPars: public PSCalib::CalibPars, pdscalibdata::PrincetonBaseV1  {
public:

  /**
   *  @brief DEPRICATED constructor, which use string& source
   *  
   *  @param[in] calibDir   Calibration directory for current experiment.
   *  @param[in] groupName  Data type and group names.
   *  @param[in] source     The name of the data source.
   *  @param[in] runNumber  Run number to search the valid file name.
   *  @param[in] print_bits =0-print no messages; +1-input parameters, +2-print msges from  PSCalib::CalibFileFinder, +4-use default, +8-missing type
   */ 
  PrincetonCalibPars ( const std::string&   calibDir,  //  /reg/d/psdm/AMO/amoa1214/calib
                       const std::string&   groupName, //  PNCCD::CalibV1
                       const std::string&   source,    //  Camp.0:pnCCD.0
                       const unsigned long& runNumber, //  7
                       unsigned             print_bits=255 );

  /**
   *  @brief Regular constructor, which use Pds::Src& src
   *  
   *  @param[in] calibDir   Calibration directory for current experiment.
   *  @param[in] groupName  Data type and group names.
   *  @param[in] src        The data source object, for example Pds::Src m_src; defined in the env.get(...,&m_src)
   *  @param[in] runNumber  Run number to search the valid file name.
   *  @param[in] print_bits =0-print no messages; +1-input parameters, +2-print msges from  PSCalib::CalibFileFinder, +4-use default, +8-missing type
   */ 
  PrincetonCalibPars ( const std::string&   calibDir,  //  /reg/d/psdm/AMO/amoa1214/calib
                       const std::string&   groupName, //  PNCCD::CalibV1
                       const Pds::Src&      src,       //  Pds::Src m_src; <- is defined in env.get(...,&m_src)
                       const unsigned long& runNumber, //  7
                       unsigned             print_bits=255 ) ;  

  virtual ~PrincetonCalibPars () ;

  /// INTERFACE METHODS

  virtual const size_t   ndim() { return pdscalibdata::PrincetonBaseV1::Ndim; };
  virtual const size_t   size() { return pdscalibdata::PrincetonBaseV1::Size; };
  virtual const shape_t* shape(){ return pdscalibdata::PrincetonBaseV1::shape_base(); };

  virtual const CalibPars::pedestals_t*    pedestals(); //{ return m_pedestals -> pedestals().data(); };
  virtual const CalibPars::pixel_gain_t*   pixel_gain();
  virtual const CalibPars::pixel_rms_t*    pixel_rms();
  virtual const CalibPars::pixel_status_t* pixel_status();
  virtual const CalibPars::common_mode_t*  common_mode();

  virtual void printCalibPars();

  /// ADDITIONAL METHODS

  void printInputPars();
  void printCalibParsStatus();

  const ndarray<const CalibPars::pedestals_t,    2> pedestals_ndarr   (){ return m_pedestals    -> get_ndarray(); };
  const ndarray<const CalibPars::pixel_status_t, 2> pixel_status_ndarr(){ return m_pixel_status -> get_ndarray(); };
  const ndarray<const CalibPars::pixel_gain_t,   2> pixel_gain_ndarr  (){ return m_pixel_gain   -> get_ndarray(); };
  const ndarray<const CalibPars::pixel_rms_t,    2> pixel_rms_ndarr   (){ return m_pixel_rms    -> get_ndarray(); };
  const ndarray<const CalibPars::common_mode_t,  1> common_mode_ndarr (){ return m_common_mode  -> get_ndarray(); };

private:

  /// Initialization, common for all constructors
  void init();

  /// Define the path to the calibration file based on input parameters
  std::string getCalibFileName(const CALIB_TYPE& calibtype);

  /// Copy constructor is disabled by default
  PrincetonCalibPars ( const PrincetonCalibPars& ) ;
  /// Assignment is disabled by default
  //PrincetonCalibPars operator = ( const PrincetonCalibPars& ) ;

//------------------
// Static Members --
//------------------

  // Data members for regular constructor 
  std::string   m_calibDir;
  std::string   m_groupName;
  std::string   m_source;
  Pds::Src      m_src;
  std::string   m_dataType;
  unsigned      m_runNumber;
  unsigned      m_print_bits;
  unsigned      m_prbits_type;
  unsigned      m_prbits_cff;
  std::string   m_name;

  // Assuming path: /reg/d/psdm/AMO/amoa1214/calib/PNCCD::CalibV1/Camp.0:pnCCD.1/pedestals/1-end.data
  std::string   m_fname;

  typedef pdscalibdata::NDArrIOV1<CalibPars::pedestals_t,2>    NDAIPEDS;
  typedef pdscalibdata::NDArrIOV1<CalibPars::pixel_gain_t,2>   NDAIGAIN;
  typedef pdscalibdata::NDArrIOV1<CalibPars::pixel_rms_t,2>    NDAIRMS;
  typedef pdscalibdata::NDArrIOV1<CalibPars::pixel_status_t,2> NDAISTATUS;
  typedef pdscalibdata::NDArrIOV1<CalibPars::common_mode_t,1>  NDAICMOD; 

  NDAIPEDS   *m_pedestals;
  NDAIGAIN   *m_pixel_gain;
  NDAIRMS    *m_pixel_rms;
  NDAISTATUS *m_pixel_status; 
  NDAICMOD   *m_common_mode;
};

} // namespace PSCalib

#endif // PSCALIB_PRINCETONCALIBPARS_H
