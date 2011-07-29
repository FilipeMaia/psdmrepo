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

/**
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
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
  CSPadCalibPars () {}

  // Test constructor
  CSPadCalibPars ( const std::string &xtc_file_name ) ;

  // Regular constructor
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

  float getCenterX(size_t quad, size_t sect){ return m_center -> getCenterX(quad,sect); };
  float getCenterY(size_t quad, size_t sect){ return m_center -> getCenterY(quad,sect); };
  float getCenterZ(size_t quad, size_t sect){ return m_center -> getCenterZ(quad,sect); };

  float getCenterCorrX(size_t quad, size_t sect){ return m_center_corr -> getCenterCorrX(quad,sect); };
  float getCenterCorrY(size_t quad, size_t sect){ return m_center_corr -> getCenterCorrY(quad,sect); };
  float getCenterCorrZ(size_t quad, size_t sect){ return m_center_corr -> getCenterCorrZ(quad,sect); };

  float getQuadMargX () { return m_marg_gap_shift -> getQuadMargX ();};
  float getQuadMargY () { return m_marg_gap_shift -> getQuadMargY ();};
  float getQuadMargZ () { return m_marg_gap_shift -> getQuadMargZ ();};

  float getMargX () { return m_marg_gap_shift -> getMargX ();};
  float getMargY () { return m_marg_gap_shift -> getMargY ();};
  float getMargZ () { return m_marg_gap_shift -> getMargZ ();};
					                 
  float getGapX  () { return m_marg_gap_shift -> getGapX  ();};
  float getGapY  () { return m_marg_gap_shift -> getGapY  ();};
  float getGapZ  () { return m_marg_gap_shift -> getGapZ  ();};
					                 
  float getShiftX() { return m_marg_gap_shift -> getShiftX(); };
  float getShiftY() { return m_marg_gap_shift -> getShiftY(); };
  float getShiftZ() { return m_marg_gap_shift -> getShiftZ(); };

  float getOffsetX(size_t quad) { return m_offset -> getOffsetX(quad); };
  float getOffsetY(size_t quad) { return m_offset -> getOffsetY(quad); };
  float getOffsetZ(size_t quad) { return m_offset -> getOffsetZ(quad); };

  float getOffsetCorrX(size_t quad) { return m_offset_corr -> getOffsetCorrX(quad); };
  float getOffsetCorrY(size_t quad) { return m_offset_corr -> getOffsetCorrY(quad); };
  float getOffsetCorrZ(size_t quad) { return m_offset_corr -> getOffsetCorrZ(quad); };

  float getRotation(size_t quad, size_t sect) { return m_rotation -> getRotation(quad,sect); };
  float getTilt    (size_t quad, size_t sect) { return m_tilt     -> getTilt    (quad,sect); };

  float getQuadRotation(size_t quad) { return m_quad_rotation -> getQuadRotation(quad); };
  float getQuadTilt    (size_t quad) { return m_quad_tilt     -> getQuadTilt    (quad); };

  static float getRowSize_um()   { return 109.92; }  // pixel size of the row in um                                           
  static float getColSize_um()   { return 109.92; }  // pixel size of the column in um                                        
  static float getGapRowSize_um(){ return 274.80; }  // pixel size of the gap column in um
  static float getGapSize_um()   { return 2*getGapRowSize_um() - getRowSize_um(); }  // pixel size of the total gap in um 
  static float getOrtSize_um()   { return 500.00; }  // pixel size of the ortogonal dimension in um                                        

  static float getRowUmToPix()   { return 1./getRowSize_um(); } // conversion factor of um to pixels for rows
  static float getColUmToPix()   { return 1./getColSize_um(); } // conversion factor of um to pixels for columns 
  static float getOrtUmToPix()   { return 1.; }                 // conversion factor of um to pixels for ort

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
  std::vector<float>       v_parameters;

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
