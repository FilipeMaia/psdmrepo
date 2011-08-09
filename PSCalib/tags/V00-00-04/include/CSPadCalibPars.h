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
#include <fstream>  // open, close etc.

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_psana/cspad.ddl.h"

#include "pdscalibdata/CalibParsCenterV1.h"      
#include "pdscalibdata/CalibParsCenterCorrV1.h"  
#include "pdscalibdata/CalibParsMargGapShiftV1.h"
#include "pdscalibdata/CalibParsOffsetV1.h"      
#include "pdscalibdata/CalibParsOffsetCorrV1.h"  
#include "pdscalibdata/CalibParsRotationV1.h"    
#include "pdscalibdata/CalibParsTiltV1.h"        
#include "pdscalibdata/CalibParsQuadRotationV1.h"
#include "pdscalibdata/CalibParsQuadTiltV1.h"    

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
 *  @brief CSPadCalibPars class loads/holds/provides access to the CSPad
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
 */

//----------------

//enum { NQuad = Psana::CsPad::MaxQuadsPerSensor};
//enum { NSect = Psana::CsPad::SectorsPerQuad};

//----------------

class CSPadCalibPars  {
public:

  // Default constructor
  // Test constructor
  CSPadCalibPars () ;

  // Regular constructor
  /**
   *  @brief Creates object which holds the calibration parameters.
   *  
   *  Loads, holds, and provides access to all calibration types 
   *  which are necessary for CSPad pixel coordinate geometry.
   *  
   *  @param[in] calibDir       Calibration directory for current experiment.
   *  @param[in] typeGroupName  Data type and group names.
   *  @param[in] source         The name of the data source.
   *  @param[in] runNumber      Run number to search the valid file name.
   */ 
  CSPadCalibPars ( const std::string&   calibDir,           //  /reg/d/psdm/cxi/cxi35711/calib
                   const std::string&   typeGroupName,      //  CsPad::CalibV1
                   const std::string&   source,             //  CxiDs1.0:Cspad.0
                   const unsigned long& runNumber ) ;       //  10

  // Destructor
  virtual ~CSPadCalibPars () ;

  size_t   getNRows             (){ return m_nrows;   };
  size_t   getNCols             (){ return m_ncols;   };

  void fillCalibNameVector();
  void getCalibFileName   ();
  void loadCalibPars      ();
  void openCalibFile      ();
  void closeCalibFile     ();
  void readCalibPars      ();
  void printCalibPars     ();

  void fillCalibParsV1();

  double getCenterX(size_t quad, size_t sect){ return m_center -> getCenterX(quad,sect); };
  double getCenterY(size_t quad, size_t sect){ return m_center -> getCenterY(quad,sect); };
  double getCenterZ(size_t quad, size_t sect){ return m_center -> getCenterZ(quad,sect); };

  double getCenterCorrX(size_t quad, size_t sect){ return m_center_corr -> getCenterCorrX(quad,sect); };
  double getCenterCorrY(size_t quad, size_t sect){ return m_center_corr -> getCenterCorrY(quad,sect); };
  double getCenterCorrZ(size_t quad, size_t sect){ return m_center_corr -> getCenterCorrZ(quad,sect); };

  double getQuadMargX () { return m_marg_gap_shift -> getQuadMargX ();};
  double getQuadMargY () { return m_marg_gap_shift -> getQuadMargY ();};
  double getQuadMargZ () { return m_marg_gap_shift -> getQuadMargZ ();};

  double getMargX () { return m_marg_gap_shift -> getMargX ();};
  double getMargY () { return m_marg_gap_shift -> getMargY ();};
  double getMargZ () { return m_marg_gap_shift -> getMargZ ();};
					                 
  double getGapX  () { return m_marg_gap_shift -> getGapX  ();};
  double getGapY  () { return m_marg_gap_shift -> getGapY  ();};
  double getGapZ  () { return m_marg_gap_shift -> getGapZ  ();};
					                 
  double getShiftX() { return m_marg_gap_shift -> getShiftX(); };
  double getShiftY() { return m_marg_gap_shift -> getShiftY(); };
  double getShiftZ() { return m_marg_gap_shift -> getShiftZ(); };

  double getOffsetX(size_t quad) { return m_offset -> getOffsetX(quad); };
  double getOffsetY(size_t quad) { return m_offset -> getOffsetY(quad); };
  double getOffsetZ(size_t quad) { return m_offset -> getOffsetZ(quad); };

  double getOffsetCorrX(size_t quad) { return m_offset_corr -> getOffsetCorrX(quad); };
  double getOffsetCorrY(size_t quad) { return m_offset_corr -> getOffsetCorrY(quad); };
  double getOffsetCorrZ(size_t quad) { return m_offset_corr -> getOffsetCorrZ(quad); };

  double getRotation(size_t quad, size_t sect) { return m_rotation -> getRotation(quad,sect); };
  double getTilt    (size_t quad, size_t sect) { return m_tilt     -> getTilt    (quad,sect); };

  double getQuadRotation(size_t quad) { return m_quad_rotation -> getQuadRotation(quad); };
  double getQuadTilt    (size_t quad) { return m_quad_tilt     -> getQuadTilt    (quad); };

  static double getRowSize_um()   { return 109.92; }  // pixel size of the row in um                                           
  static double getColSize_um()   { return 109.92; }  // pixel size of the column in um                                        
  static double getGapRowSize_um(){ return 274.80; }  // pixel size of the gap column in um
  static double getGapSize_um()   { return 2*getGapRowSize_um() - getRowSize_um(); }  // pixel size of the total gap in um 
  static double getOrtSize_um()   { return 500.00; }  // pixel size of the ortogonal dimension in um                                        

  static double getRowUmToPix()   { return 1./getRowSize_um(); } // conversion factor of um to pixels for rows
  static double getColUmToPix()   { return 1./getColSize_um(); } // conversion factor of um to pixels for columns 
  static double getOrtUmToPix()   { return 1.; }                 // conversion factor of um to pixels for ort

private:

  // Copy constructor and assignment are disabled by default
  CSPadCalibPars ( const CSPadCalibPars& ) ;
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
  std::string   m_dataType;
  unsigned long m_runNumber;

  std::vector<std::string> v_calibname; // center, center_corr, off_gap_shift, offset, offset_corr, rotation, tilt, ...
  std::vector<double>      v_parameters;

  std::string m_cur_calibname;  
  std::string m_fname;

  bool m_isTestMode;

  size_t m_nrows; 
  size_t m_ncols; 

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

};

} // namespace PSCalib

#endif // PSCALIB_CSPADCALIBPARS_H
