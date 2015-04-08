#ifndef PSCALIB_CALIBFILEFINDER_H
#define PSCALIB_CALIBFILEFINDER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// $Revision$
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Src.hh"

//-----------------------------

namespace PSCalib {

/// @addtogroup PSCalib PSCalib

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
   *  Calibration directory typically comes from environment for psana jobs.
   *
   *  @param[in] calibDir   Calibration directory for current experiment.
   *  @param[in] className  Calibration class name, e.g. CsPad::CalibV1
   *  @param[in] print_bits =0-print nothing, +1-wrong file extension, +2-skipping file
   */
    
  CalibFileFinder (const std::string& calibDir,
                   const std::string& className,
                   const unsigned& print_bits=255);

  // Destructor
  ~CalibFileFinder () ;

  /**
   *  @brief Returns complete path/name of the calibration file.
   *
   *  @param[in] src        The name of the data source, e.g. CxiDs1.0:Cspad.0
   *  @param[in] datatype   Type of the calibration parameters (i.e. "rotation").
   *  @param[in] runNumber  Run number to search the valid file name.
   */
  std::string findCalibFile(const std::string& src, const std::string& datatype, unsigned long runNumber) const;

  /**
   *  @brief Returns complete path/name of the calibration file.
   *
   *  @param[in] src        Address of the data source, only DetInfo addresses are accepted.
   *  @param[in] datatype   Type of the calibration parameters (i.e. "rotation").
   *  @param[in] runNumber  Run number to search the valid file name.
   */
  std::string findCalibFile(const Pds::Src& src, const std::string& datatype, unsigned long runNumber) const;
 
  /**
   *  @brief Selects calibration file from a list of file names.
   *
   *  This method is mostly for testing purposes, it is used in implementation of findCalibFile().
   *  It can be used if you have the list of file names instead of scanning pre-defined directory.
   *  File names that do not match standard naming convention are ignored. Standard convention is
   *  two run number separated with '-' and extension '.data', second run can be specified as 'end'.
   *
   *  @param[in] files      List of file names.
   *  @param[in] runNumber  Run number to search the valid file name.
   *  @param[in] print_bits print control bit-word.
   */
  static std::string selectCalibFile(const std::vector<std::string>& files, unsigned long runNumber, unsigned print_bits=255);


  /**
   *  @brief If source name has DetInfo(...) - remove it
   *  @param[in] str - input string with source name, ex: "DetInfo(Camp.0:pnCCD.0)"
   */
  static std::string trancateSourceName(const std::string& str);


protected:

private:

  // Data members
  const std::string m_calibDir;
  const std::string m_typeGroupName;
  unsigned          m_print_bits;
};

} // namespace PSCalib

#endif // PSCALIB_CALIBFILEFINDER_H
