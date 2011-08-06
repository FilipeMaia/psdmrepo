#ifndef PSCALIB_CALIBFILEFINDER_H
#define PSCALIB_CALIBFILEFINDER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CalibFileFinder.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSCalib {

/**
 *  @ingroup PSCalib  
 *
 *  @brief CalibFileFinder class finds the pass to calibration file.
 *
 *  When all input parameters are provided at class initialization
 *  the method findCalibFile(...) returns the path/name to the file 
 *  with requested calibration parameters.
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see CSPadCalibPars
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class CalibFileFinder  {
public:

  // Default constructor
  CalibFileFinder () {}

  // Default constructor

  /**
   *  @brief Creates object with elements of the path to the calibration file.
   *  
   *  @param[in] calibDir       Calibration directory for current experiment.
   *  @param[in] typeGroupName  Data type and group names.
   *  @param[in] src            The name of the data source.
   */ 

  CalibFileFinder (const std::string& calibDir,          //  /reg/d/psdm/cxi/cxi35711/calib
                   const std::string& typeGroupName,     //  CsPad::CalibV1
                   const std::string& src);              //  CxiDs1.0:Cspad.0) ;

  // Destructor
  virtual ~CalibFileFinder () ;

  // find calibration file 
  /**
   *  @brief Returns complete path/name of the calibration file.
   *  
   *  @param[in] datatype   Type of the calibration parameters (i.e. "rotation").
   *  @param[in] runNumber  Run number to search the valid file name.
   */ 
  std::string findCalibFile(const std::string& datatype, unsigned long& runNumber) const;
 
protected:

private:

  // Data members
  
  const std::string m_calibDir;
  const std::string m_typeGroupName;
  const std::string m_src;
  const std::string m_dataType;

  // Copy constructor and assignment are disabled by default
  CalibFileFinder ( const CalibFileFinder& ) ;
  CalibFileFinder& operator = ( const CalibFileFinder& ) ;

};

} // namespace PSCalib

#endif // PSCALIB_CALIBFILEFINDER_H
