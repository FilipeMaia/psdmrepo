#ifndef PSCALIB_CSPAD2X2CALIBPARS_H
#define PSCALIB_CSPAD2X2CALIBPARS_H

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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/cspad.ddl.h"
#include "pdsdata/xtc/Src.hh"

#include "pdscalibdata/CsPad2x2CenterV1.h"      
#include "pdscalibdata/CsPad2x2TiltV1.h"        

//-----------------------------

namespace PSCalib {

/// @addtogroup PSCalib PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief CSPad2x2CalibPars class loads/holds/provides access to the CSPad2x2
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
 *  #include "PSCalib/CSPad2x2CalibPars.h"
 *  typedef PSCalib::CSPad2x2CalibPars CALIB;
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
 *  const std::string calibDir   = "/reg/d/psdm/xpp/xpptut13/calib/";
 *  const std::string groupName  = "CsPad2x2::CalibV1/";
 *  unsigned long     runNumber  = 10;
 *  Pds::Src src; env.get(...,&src);
 *  CALIB *calibpars = new CALIB(calibDir, groupName, src, runNumber);  
 *  @endcode
 *  \n
 *  For explicit constructor (depricated):
 *  @code
 *  const std::string calibDir   = "/reg/d/psdm/xpp/xpptut13/calib/";
 *  const std::string groupName  = "CsPad2x2::CalibV1/";
 *  const std::string source     = "XppGon.0:Cspad2x2.1";
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
 *  int status  = calibpars -> getCalibTypeStatus("center") // Returns status: 0-default, 1-loaded from file
 *  double xc   = calibpars -> getCenterX(sect);
 *  double tilt = calibpars -> getTilt(sect);
 *  ... etc. for all other access methods
 *  @endcode
 *
 *  @see CalibFileFinder
 */

//----------------

class CSPad2x2CalibPars  {
public:

  /// Default and test constructor
  CSPad2x2CalibPars ( bool isTestMode = false ) ;


  /**
   *  @brief DEPRICATED constructor, which use string& source
   *  
   *  @param[in] calibDir       Calibration directory for current experiment.
   *  @param[in] typeGroupName  Data type and group names.
   *  @param[in] source         The name of the data source.
   *  @param[in] runNumber      Run number to search the valid file name.
   */ 
  CSPad2x2CalibPars ( const std::string&   calibDir,           //  /reg/d/psdm/mec/mec73313/calib
                      const std::string&   typeGroupName,      //  CsPad2x2::CalibV1
                      const std::string&   source,             //  MecTargetChamber.0:Cspad2x2.1
                      const unsigned long& runNumber ) ;       //  10

  /**
   *  @brief Regular constructor, which use Pds::Src& src
   *  
   *  @param[in] calibDir       Calibration directory for current experiment.
   *  @param[in] typeGroupName  Data type and group names.
   *  @param[in] src            The data source object, for example Pds::Src m_src; defined in the env.get(...,&m_src)
   *  @param[in] runNumber      Run number to search the valid file name.
   */ 
  CSPad2x2CalibPars ( const std::string&   calibDir,           //  /reg/d/psdm/mec/mec73313/calib
                      const std::string&   typeGroupName,      //  CsPad2x2::CalibV1
                      const Pds::Src&      src,                //  Pds::Src m_src; <- is defined in env.get(...,&m_src)
                      const unsigned long& runNumber ) ;       //  10

  /// Destructor
  virtual ~CSPad2x2CalibPars () ;

  //size_t   getNRows             (){ return m_nrows;   };
  //size_t   getNCols             (){ return m_ncols;   };

  /// Makes member data vector with all supported calibration types such as center, tilt, ...
  void fillCalibNameVector   ();

  /// Define the path to the calibration file based on input parameters
  void getCalibFileName      ();

  /// Load all known calibration parameters
  void loadCalibPars         ();

  /// Open calibration file
  void openCalibFile         ();

  /// Close calibration file
  void closeCalibFile        ();

  /// Read parameters from calibration file to vector
  void readCalibPars         ();

  /// Fill calibration parameters from vector
  void fillCalibParsV1       ();

  /// Fill default calibration parameters
  void fillDefaultCalibParsV1();

  /// Generate error message in the log and abort
  void fatalMissingFileName  ();

  /// Generate warning message in the log
  void msgUseDefault         ();

  /// Prints calibration parameters
  void printCalibPars        ();

  /// Prints input parameters of the object
  void printInputPars        ();

  /// Prints calibration parameters status
  void printCalibParsStatus  ();

  /// Returns x-coordinate [pix] of the 2x1 section center @param[in] sect - 2x1 section number 0 or 1
  double getCenterX(size_t sect){ return m_center -> getCenterX(sect); };

  /// Returns y-coordinate [pix] of the 2x1 section center @param[in] sect - 2x1 section number 0 or 1 
  double getCenterY(size_t sect){ return m_center -> getCenterY(sect); };

  /// Returns z-coordinate [pix] of the 2x1 section center @param[in] sect - 2x1 section number 0 or 1 
  double getCenterZ(size_t sect){ return m_center -> getCenterZ(sect); };

  /// Returns the tilt angle [degree] of the 2x1 section @param[in] sect - 2x1 section number 0 or 1
  double getTilt   (size_t sect){ return m_tilt   -> getTilt   (sect); };

  /// Returns 109.92 um
  static double getRowSize_um()   { return 109.92; }  // pixel size of the row in um                                           

  /// Returns 109.92 um
  static double getColSize_um()   { return 109.92; }  // pixel size of the column in um                                        

  /// Returns 274.80 um
  static double getGapRowSize_um(){ return 274.80; }  // pixel size of the gap column in um

  /// Returns size of the gap
  static double getGapSize_um()   { return 2*getGapRowSize_um() - getRowSize_um(); }  // pixel size of the total gap in um 

  /// Returns 500 um
  static double getOrtSize_um()   { return 500.00; }  // pixel size of the ortogonal dimension in um  

  /// Returns 1 / 109.92 um
  static double getRowUmToPix()   { return 1./getRowSize_um(); } // conversion factor of um to pixels for rows

  /// Returns 1 / 109.92 um
  static double getColUmToPix()   { return 1./getColSize_um(); } // conversion factor of um to pixels for columns 

  /// Returns 1
  static double getOrtUmToPix()   { return 1.; }                 // conversion factor of um to pixels for ort

  /// Returns status of the calibration constants, 0-default, 1-loaded from file @param[in] type - calibration type string-name, for example "center" or "tilt"
  int getCalibTypeStatus(const std::string&  type) { return m_calibtype_status[type]; };

private:

  /// Copy constructor is disabled by default
  CSPad2x2CalibPars ( const CSPad2x2CalibPars& ) ;
  /// Assignment is disabled by default
  CSPad2x2CalibPars operator = ( const CSPad2x2CalibPars& ) ;

//------------------
// Static Members --
//------------------

  // Assuming path: /reg/d/psdm/mec/mec73313/calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/1-end.data
  // Data members for TEST constructor       
  std::string m_calibdir;       // /reg/d/psdm/mec/mec73313/calib
  std::string m_calibfilename;  // 1-end.data

  // Data members for regular constructor 
  std::string   m_calibDir;
  std::string   m_typeGroupName;
  std::string   m_source;
  Pds::Src      m_src;
  std::string   m_dataType;
  unsigned long m_runNumber;

  std::vector<std::string> v_calibname; // center, tilt, ...
  std::vector<double>      v_parameters;

  std::map<std::string, int> m_calibtype_status; // =0-default, =1-from file

  std::string m_cur_calibname;  
  std::string m_fname;

  bool m_isTestMode;

  //size_t m_nrows; 
  //size_t m_ncols; 

  std::ifstream m_file;

  pdscalibdata::CsPad2x2CenterV1 *m_center;
  pdscalibdata::CsPad2x2TiltV1   *m_tilt;   
};

} // namespace PSCalib

#endif // PSCALIB_CSPAD2X2CALIBPARS_H
