#ifndef PSCALIB_GENERICCALIBPARS_H
#define PSCALIB_GENERICCALIBPARS_H

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
#include "pdscalibdata/NDArrIOV1.h"      

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
#include "pdsdata/xtc/Src.hh"

//-------------------------------

namespace PSCalib {

/// @addtogroup PSCalib PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief GenericCalibPars class loads/holds/provides access to the pnCCD
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
 *  #include "PSCalib/GenericCalibPars.h"
 *
 *  #include "pdscalibdata/CsPadBaseV2.h"
 *  #include "pdscalibdata/CsPad2x2BaseV2.h"
 *  #include "pdscalibdata/PnccdBaseV1.h"
 *  #include "pdscalibdata/PrincetonBaseV1.h"
 *  #include "pdscalibdata/AndorBaseV1.h"
 *  #include "pdscalibdata/Opal1000BaseV1.h"
 *  #include "pdscalibdata/Opal4000BaseV1.h"
 *  ...
 *
 *  typedef PSCalib::GenericCalibPars<pdscalibdata::PnccdBaseV2>  CALIB;
 *  typedef PSCalib::GenericCalibPars<pdscalibdata::...BaseV2> ...CALIB;
 *  ... 
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
 *  unsigned          print_bits = 255;
 *  Pds::Src src; env.get(source, key, &src);
 *  CALIB *calibpars = new CALIB(calibDir, groupName, src, runNumber, print_bits);  
 *  @endcode
 *  \n
 *  or similar for explicit constructor (depricated):
 *  @code
 *  CALIB *calibpars = new CALIB(calibDir, groupName, source, runNumber, print_bits);  
 *  @endcode
 *  \n
 *  or for the same list of parameters using polymorphism:
 *  @code
 *  PSCalib::CalibPars *calibpars = new CALIB(calibDir, groupName, source, runNumber, print_bits);  
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
template <typename TBASE> // TBASE stands for something like pdscalibdata::PrincetonBaseV1
class GenericCalibPars: public PSCalib::CalibPars, TBASE  {

public:


  typedef PSCalib::CalibPars::shape_t shape_t;


    // const static size_t Ndim = TBASE::Ndim; 

  /**
   *  @brief DEPRICATED constructor, which use string& source
   *  
   *  @param[in] calibDir   Calibration directory for current experiment.
   *  @param[in] groupName  Data type and group names.
   *  @param[in] source     The name of the data source.
   *  @param[in] runNumber  Run number to search the valid file name.
   *  @param[in] print_bits =0-print no messages; +1-input parameters, +2-print msges from  PSCalib::CalibFileFinder, +4-use default, +8-missing type
   */ 
  GenericCalibPars ( const std::string&   calibDir,  //  /reg/d/psdm/AMO/amoa1214/calib
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
  GenericCalibPars ( const std::string&   calibDir,  //  /reg/d/psdm/AMO/amoa1214/calib
                     const std::string&   groupName, //  PNCCD::CalibV1
                     const Pds::Src&      src,       //  Pds::Src m_src; <- is defined in env.get(...,&m_src)
                     const unsigned long& runNumber, //  7
                     unsigned             print_bits=255 ) ;  

  virtual ~GenericCalibPars () ;

  /// INTERFACE METHODS

  virtual const size_t   ndim() { return TBASE::Ndim; }
  virtual const size_t   size();
  virtual const shape_t* shape();

  virtual const CalibPars::pedestals_t*    pedestals(); 
  virtual const CalibPars::pixel_gain_t*   pixel_gain();
  virtual const CalibPars::pixel_rms_t*    pixel_rms();
  virtual const CalibPars::pixel_status_t* pixel_status();
  virtual const CalibPars::common_mode_t*  common_mode();

  virtual void printCalibPars();

  /// ADDITIONAL METHODS

  const size_t   size_of_ndarray();
  const shape_t* shape_of_ndarray();


  void printInputPars();
  void printCalibParsStatus();
  std::string str_shape();

  // What if 0-pointer?
  //const ndarray<const CalibPars::pedestals_t,    TBASE::Ndim> pedestals_ndarr   (){ return m_pedestals    -> get_ndarray(); }
  //const ndarray<const CalibPars::pixel_status_t, TBASE::Ndim> pixel_status_ndarr(){ return m_pixel_status -> get_ndarray(); }
  //const ndarray<const CalibPars::pixel_gain_t,   TBASE::Ndim> pixel_gain_ndarr  (){ return m_pixel_gain   -> get_ndarray(); }
  //const ndarray<const CalibPars::pixel_rms_t,    TBASE::Ndim> pixel_rms_ndarr   (){ return m_pixel_rms    -> get_ndarray(); }
  //const ndarray<const CalibPars::common_mode_t,            1> common_mode_ndarr (){ return m_common_mode  -> get_ndarray(); }

private:

  /// Initialization, common for all constructors
  void init();

  /// Define the path to the calibration file based on input parameters
  std::string getCalibFileName(const CALIB_TYPE& calibtype);

  /// Copy constructor is disabled by default
  GenericCalibPars ( const GenericCalibPars& ) ;
  /// Assignment is disabled by default
  //GenericCalibPars operator = ( const GenericCalibPars& ) ;

  /// Request to load all calib pars for printCalibParsStatus() and printCalibPars()
  void loadAllCalibPars();

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

  typedef pdscalibdata::NDArrIOV1<CalibPars::pedestals_t,   TBASE::Ndim> NDAIPEDS;
  typedef pdscalibdata::NDArrIOV1<CalibPars::pixel_gain_t,  TBASE::Ndim> NDAIGAIN;
  typedef pdscalibdata::NDArrIOV1<CalibPars::pixel_rms_t,   TBASE::Ndim> NDAIRMS;
  typedef pdscalibdata::NDArrIOV1<CalibPars::pixel_status_t,TBASE::Ndim> NDAISTATUS;
  typedef pdscalibdata::NDArrIOV1<CalibPars::common_mode_t,           1> NDAICMOD; 

  NDAIPEDS   *m_pedestals;
  NDAIGAIN   *m_pixel_gain;
  NDAIRMS    *m_pixel_rms;
  NDAISTATUS *m_pixel_status; 
  NDAICMOD   *m_common_mode;
};

} // namespace PSCalib

#endif // PSCALIB_GENERICCALIBPARS_H
