#ifndef PSCALIB_CSPAD2X2CALIBPARS_H
#define PSCALIB_CSPAD2X2CALIBPARS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPad2x2CalibPars.
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
#include "pdsdata/xtc/Src.hh"

#include "pdscalibdata/CsPad2x2CenterV1.h"      
#include "pdscalibdata/CsPad2x2TiltV1.h"        

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
 */

//----------------

class CSPad2x2CalibPars  {
public:

  /// Default and test constructor
  CSPad2x2CalibPars ( bool isTestMode = false ) ;

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

  // DEPRICATED constructor, which use string& source ...
  CSPad2x2CalibPars ( const std::string&   calibDir,           //  /reg/d/psdm/mec/mec73313/calib
                      const std::string&   typeGroupName,      //  CsPad2x2::CalibV1
                      const std::string&   source,             //  MecTargetChamber.0:Cspad2x2.1
                      const unsigned long& runNumber ) ;       //  10

  CSPad2x2CalibPars ( const std::string&   calibDir,           //  /reg/d/psdm/mec/mec73313/calib
                      const std::string&   typeGroupName,      //  CsPad2x2::CalibV1
                      const Pds::Src&      src,                //  Pds::Src m_src; <- is defined in env.get(...,&m_src)
                      const unsigned long& runNumber ) ;       //  10

  // Destructor
  virtual ~CSPad2x2CalibPars () ;

  size_t   getNRows             (){ return m_nrows;   };
  size_t   getNCols             (){ return m_ncols;   };

  void fillCalibNameVector   ();
  void getCalibFileName      ();
  void loadCalibPars         ();
  void openCalibFile         ();
  void closeCalibFile        ();
  void readCalibPars         ();
  void fillCalibParsV1       ();
  void fillDefaultCalibParsV1();
  void fatalMissingFileName  ();
  void msgUseDefault         ();
  void printCalibPars        ();
  void printInputPars        ();


  double getCenterX(size_t sect){ return m_center -> getCenterX(sect); };
  double getCenterY(size_t sect){ return m_center -> getCenterY(sect); };
  double getCenterZ(size_t sect){ return m_center -> getCenterZ(sect); };

  double getTilt   (size_t sect){ return m_tilt   -> getTilt   (sect); };

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
  CSPad2x2CalibPars ( const CSPad2x2CalibPars& ) ;
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

  std::string m_cur_calibname;  
  std::string m_fname;

  bool m_isTestMode;

  size_t m_nrows; 
  size_t m_ncols; 

  std::ifstream m_file;

  pdscalibdata::CsPad2x2CenterV1 *m_center;
  pdscalibdata::CsPad2x2TiltV1   *m_tilt;   
};

} // namespace PSCalib

#endif // PSCALIB_CSPAD2X2CALIBPARS_H
