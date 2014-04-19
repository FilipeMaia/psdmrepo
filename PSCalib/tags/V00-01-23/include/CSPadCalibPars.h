#ifndef PSCALIB_CSPADCALIBPARS_H
#define PSCALIB_CSPADCALIBPARS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadCalibPars.
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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/cspad.ddl.h"
#include "pdsdata/xtc/Src.hh"

#include "pdscalibdata/CalibParsCenterV1.h"      
#include "pdscalibdata/CalibParsCenterCorrV1.h"  
#include "pdscalibdata/CalibParsMargGapShiftV1.h"
#include "pdscalibdata/CalibParsOffsetV1.h"      
#include "pdscalibdata/CalibParsOffsetCorrV1.h"  
#include "pdscalibdata/CalibParsRotationV1.h"    
#include "pdscalibdata/CalibParsTiltV1.h"        
#include "pdscalibdata/CalibParsQuadRotationV1.h"
#include "pdscalibdata/CalibParsQuadTiltV1.h"    
#include "pdscalibdata/CsPadBeamVectorV1.h"    
#include "pdscalibdata/CsPadBeamIntersectV1.h"    
#include "pdscalibdata/CsPadCenterGlobalV1.h"      
#include "pdscalibdata/CsPadRotationGlobalV1.h"    

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSCalib {

/**
 *  @defgroup PSCalib PSCalib package
 *  @brief Package PSCalib provides access to the CSPAD calibration parameters
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 */

///  @addtogroup PSCalib PSCalib
 
/**
 *
 *  @ingroup PSCalib
 *
 *  @brief CSPadCalibPars class loads/holds/provides access to the CSPad
 *  geometry calibration parameters.
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Include and typedef
 *  @code
 *  #include "PSCalib/CSPadCalibPars.h"
 *  typedef PSCalib::CSPadCalibPars CALIB;
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
 *  const std::string calibDir   = "/reg/d/psdm/cxi/cxitut13/calib";
 *  const std::string groupName  = "CsPad::CalibV1";
 *  unsigned long     runNumber  = 10;
 *  Pds::Src src; env.get(...,&src);
 *  CALIB *calibpars = new CALIB(calibDir, groupName, src, runNumber);  
 *  @endcode
 *  \n
 *  For explicit constructor (depricated):
 *  @code
 *  const std::string calibDir   = "/reg/d/psdm/cxi/cxitut13/calib";
 *  const std::string groupName  = "CsPad::CalibV1";
 *  const std::string source     = "CxiDs1.0:Cspad.0";
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
 *  size_t quad=1, sect=7; // for example...
 *  int status  = calibpars -> getCalibTypeStatus("center_global") // Returns status: 0-default, 1-loaded from file
 *  double xc   = calibpars -> getCenterX(quad, sect);
 *  double xcg  = calibpars -> getCenterGlobalX(quad, sect);
 *  double tilt = calibpars -> getTilt(quad, sect);
 *  ... etc. for all other access methods
 *  @endcode
 *
 *  @see CalibFileFinder
 *
 */

//----------------

class CSPadCalibPars  {
public:

  /**
   *  @brief Default and test constructor
   *  @param[in] isTestMode - flag of the test mode; if =true then parameters are loaded from the test directory.
   */ 
  CSPadCalibPars ( bool isTestMode = false ) ;

  /**
   *  @brief DEPRICATED constructor, which gets the source as a string\&. It is preserved for backward compatability.
   *  @param[in] calibDir       Calibration directory for current experiment, for example "/reg/d/psdm/cxi/cxitut13/calib";
   *  @param[in] typeGroupName  Data type and group names, for example "CsPad::CalibV1";
   *  @param[in] source         The name of the data source, for example "CxiDs1.0:Cspad.0";
   *  @param[in] runNumber      Run number to search the valid file name, for example  =10;
   */ 
  CSPadCalibPars ( const std::string&   calibDir,     
                   const std::string&   typeGroupName,
                   const std::string&   source,       
                   const unsigned long& runNumber ) ;

  /**
   *  @brief RECOMMENDED constructor, which gets the source as a \c const \c Pds::Src\& parameter.
   *  @param[in] calibDir       Calibration directory for current experiment, for example "/reg/d/psdm/cxi/cxitut13/calib";
   *  @param[in] typeGroupName  Data type and group names, for example "CsPad::CalibV1";
   *  @param[in] src            The name of the data source, for example Pds::Src m_src; defined in the env.get(...,&m_src)
   *  @param[in] runNumber      Run number to search the valid file name, for example  =10;
   */ 
  CSPadCalibPars ( const std::string&   calibDir,     
                   const std::string&   typeGroupName,
                   const Pds::Src&      src,          
                   const unsigned long& runNumber ) ;

  /// Destructor
  virtual ~CSPadCalibPars () ;

  // Returns the number of rows in 2x1 (185)          
  //size_t   getNRows          (){ return m_nrows; };

  // Returns the number of columns in 2x1 (388)
  //size_t   getNCols          (){ return m_ncols; };

  /// Prints status for all calibration parameters: 0-default, 1-loaded from file
  void printCalibParsStatus  ();

  /// Prints all calibration parameters
  void printCalibPars        ();

  /// Prints input parameters
  void printInputPars        ();



  /// Returns x-coordinate [pix] of the 2x1 section center in specified quad.
  double getCenterX(size_t quad, size_t sect){ return m_center -> getCenterX(quad,sect); };

  /// Returns y-coordinate [pix] of the 2x1 section center in specified quad.
  double getCenterY(size_t quad, size_t sect){ return m_center -> getCenterY(quad,sect); };

  /// Returns z-coordinate [pix] of the 2x1 section center in specified quad.
  double getCenterZ(size_t quad, size_t sect){ return m_center -> getCenterZ(quad,sect); };



  /// Returns x-coordinate correction [pix] of the 2x1 section center in specified quad.
  double getCenterCorrX(size_t quad, size_t sect){ return m_center_corr -> getCenterCorrX(quad,sect); };

  /// Returns y-coordinate correction [pix] of the 2x1 section center in specified quad.
  double getCenterCorrY(size_t quad, size_t sect){ return m_center_corr -> getCenterCorrY(quad,sect); };

  /// Returns z-coordinate correction [pix] of the 2x1 section center in specified quad.
  double getCenterCorrZ(size_t quad, size_t sect){ return m_center_corr -> getCenterCorrZ(quad,sect); };



  /// Returns x-margine of all 2x1s in quad from calibration type marg_gap_shift
  double getQuadMargX () { return m_marg_gap_shift -> getQuadMargX (); };

  /// Returns y-margine of all 2x1s in quad from calibration type marg_gap_shift
  double getQuadMargY () { return m_marg_gap_shift -> getQuadMargY (); };

  /// Returns z-margine of all 2x1s in quad from calibration type marg_gap_shift
  double getQuadMargZ () { return m_marg_gap_shift -> getQuadMargZ (); };



  /// Returns x-margine of all quads in the detector from calibration type marg_gap_shift
  double getMargX () { return m_marg_gap_shift -> getMargX (); };

  /// Returns y-margine of all quads in the detector from calibration type marg_gap_shift
  double getMargY () { return m_marg_gap_shift -> getMargY (); };

  /// Returns z-margine of all quads in the detector from calibration type marg_gap_shift
  double getMargZ () { return m_marg_gap_shift -> getMargZ (); };
					                 


  /// Returns x-gap between quads in the detector from calibration type marg_gap_shift
  double getGapX  () { return m_marg_gap_shift -> getGapX  (); };

  /// Returns y-gap between quads in the detector from calibration type marg_gap_shift
  double getGapY  () { return m_marg_gap_shift -> getGapY  (); };

  /// Returns z-gap between quads in the detector from calibration type marg_gap_shift
  double getGapZ  () { return m_marg_gap_shift -> getGapZ  (); };


					                 
  /// Returns x-shift between quads in the detector from calibration type marg_gap_shift
  double getShiftX() { return m_marg_gap_shift -> getShiftX(); };
					                 
  /// Returns y-shift between quads in the detector from calibration type marg_gap_shift
  double getShiftY() { return m_marg_gap_shift -> getShiftY(); };
					                 
  /// Returns z-shift between quads in the detector from calibration type marg_gap_shift
  double getShiftZ() { return m_marg_gap_shift -> getShiftZ(); };



  /// Returns x-offset of quads in the detector from calibration type offset
  double getOffsetX(size_t quad) { return m_offset -> getOffsetX(quad); };

  /// Returns y-offset of quads in the detector from calibration type offset
  double getOffsetY(size_t quad) { return m_offset -> getOffsetY(quad); };

  /// Returns z-offset of quads in the detector from calibration type offset
  double getOffsetZ(size_t quad) { return m_offset -> getOffsetZ(quad); };



  /// Returns x-offset correction of quads in the detector from calibration type offset_corr
  double getOffsetCorrX(size_t quad) { return m_offset_corr -> getOffsetCorrX(quad); };

  /// Returns y-offset correction of quads in the detector from calibration type offset_corr
  double getOffsetCorrY(size_t quad) { return m_offset_corr -> getOffsetCorrY(quad); };

  /// Returns z-offset correction of quads in the detector from calibration type offset_corr
  double getOffsetCorrZ(size_t quad) { return m_offset_corr -> getOffsetCorrZ(quad); };



  /// Returns the 2x1 tile rotation angle (in units of n*90 degrees) from calibration type rotation 
  double getRotation(size_t quad, size_t sect) { return m_rotation -> getRotation(quad,sect); };

  /// Returns the 2x1 tile tilt angle (in units of n*90 degrees) from calibration type tilt 
  double getTilt    (size_t quad, size_t sect) { return m_tilt     -> getTilt    (quad,sect); };



  /// Returns the quad rotation angle (in units of n*90 degrees) from calibration type quad_rotation 
  double getQuadRotation(size_t quad) { return m_quad_rotation -> getQuadRotation(quad); };

  /// Returns the quad tilt angle (in units of n*90 degrees) from calibration type quad_tilt
  double getQuadTilt    (size_t quad) { return m_quad_tilt     -> getQuadTilt    (quad); };



  /// Returns pointer to the beam_vector from calibration type beam_vector 
  double* getBeamVector   ()         { return  m_beam_vector    -> getVector(); };

  /// Returns component of the beam_vector from calibration type beam_vector 
  double  getBeamVector   (size_t i) { return  m_beam_vector    -> getVectorEl(i); };



  /// Returns pointer to the beam_intersect from calibration type beam_intersect 
  double* getBeamIntersect()         { return  m_beam_intersect -> getVector(); };

  /// Returns component of the beam_intersect from calibration type beam_intersect 
  double  getBeamIntersect(size_t i) { return  m_beam_intersect -> getVectorEl(i); };



  /// Returns the pixel size of the row in um                                           
  static double getRowSize_um()   { return 109.92; }

  /// Returns the pixel size of the column in um                                        
  static double getColSize_um()   { return 109.92; }  

  /// Returns the pixel size of the gap column in um
  static double getGapRowSize_um(){ return 274.80; }

  /// Returns the pixel size of the total gap in um 
  static double getGapSize_um()   { return 2*getGapRowSize_um() - getRowSize_um(); }

  /// Returns the pixel size of the ortogonal dimension in um                                        
  static double getOrtSize_um()   { return 500.00; }

  /// Returns the conversion factor of um to pixels for rows
  static double getRowUmToPix()   { return 1./getRowSize_um(); }
 
  /// Returns the conversion factor of um to pixels for columns 
  static double getColUmToPix()   { return 1./getColSize_um(); }

  /// Returns the conversion factor of um to pixels for ort.
  static double getOrtUmToPix()   { return 1.; }                

  /// Returns x-coordinate [pix] of the 2x1 section center for specified quad in the detector frame.
  double getCenterGlobalX(size_t quad, size_t sect){ return m_center_global -> getCenterX(quad,sect); };

  /// Returns y-coordinate [pix] of the 2x1 section center for specified quad in the detector frame.
  double getCenterGlobalY(size_t quad, size_t sect){ return m_center_global -> getCenterY(quad,sect); };

  /// Returns z-coordinate [pix] of the 2x1 section center for specified quad in the detector frame.
  double getCenterGlobalZ(size_t quad, size_t sect){ return m_center_global -> getCenterZ(quad,sect); };

  /// Returns rotation angle of the 2x1 section for specified quad in the detector frame.
  double getRotationGlobal(size_t quad, size_t sect) { return m_rotation_global -> getRotation(quad,sect); };

  /// Returns status of the calibration constants, 0-default, 1-loaded from file
  int getCalibTypeStatus(const std::string&  type) { return m_calibtype_status[type]; };


protected:

  /// Makes member data vector with all supported calibration types such as center, center_corr, off_gap_shift, offset, offset_corr, rotation, tilt, ...
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


private:

  /// Copy constructor
  CSPadCalibPars ( const CSPadCalibPars& ) ;
  /// Assignment operator
  CSPadCalibPars operator = ( const CSPadCalibPars& ) ;

//------------------
// Static Members --
//------------------

  // Data members for TEST constructor       
  //std::string m_expdir;       // /reg/d/psdm/CXI/cxi35711 
  //std::string m_calibtype;    // CsPad::CalibV1 
  //std::string m_calibsrc;     // CxiDs1.0:Cspad.0 
  std::string m_calibdir;       // /reg/neh/home/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi35711-r0009-det
  std::string m_calibfilename;  // 1-20.data


  // Data members for regular constructor // /reg/d/psdm/CXI/cxi35711/calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/pedestals/1-20.data
  std::string   m_calibDir;
  std::string   m_typeGroupName;
  std::string   m_source;
  Pds::Src      m_src;
  std::string   m_dataType;
  unsigned long m_runNumber;

  std::vector<std::string> v_calibname; // center, center_corr, off_gap_shift, offset, offset_corr, rotation, tilt, ...
  std::vector<double>      v_parameters;

  std::map<std::string, int> m_calibtype_status; // =0-default, =1-from file

  std::string m_cur_calibname;  
  std::string m_fname;

  bool m_isTestMode;

  //size_t m_nrows; 
  //size_t m_ncols; 

  std::ifstream m_file;

  pdscalibdata::CalibParsCenterV1       *m_center;
  pdscalibdata::CalibParsCenterCorrV1   *m_center_corr;
  pdscalibdata::CalibParsMargGapShiftV1 *m_marg_gap_shift;
  pdscalibdata::CalibParsOffsetV1       *m_offset;
  pdscalibdata::CalibParsOffsetCorrV1   *m_offset_corr;
  pdscalibdata::CalibParsRotationV1     *m_rotation;    
  pdscalibdata::CalibParsTiltV1         *m_tilt;   
  pdscalibdata::CalibParsQuadRotationV1 *m_quad_rotation;    
  pdscalibdata::CalibParsQuadTiltV1     *m_quad_tilt;   
  pdscalibdata::CsPadBeamVectorV1       *m_beam_vector;   
  pdscalibdata::CsPadBeamIntersectV1    *m_beam_intersect;   
  pdscalibdata::CsPadCenterGlobalV1     *m_center_global;
  pdscalibdata::CsPadRotationGlobalV1   *m_rotation_global;    

};

} // namespace PSCalib

#endif // PSCALIB_CSPADCALIBPARS_H
