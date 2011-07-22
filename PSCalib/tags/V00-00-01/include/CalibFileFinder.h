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
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
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
  CalibFileFinder (const std::string& calibDir,          //  /reg/d/psdm/cxi/cxi35711/calib
                   const std::string& typeGroupName,     //  CsPad::CalibV1
                   const std::string& src);              //  CxiDs1.0:Cspad.0) ;

  // Destructor
  virtual ~CalibFileFinder () ;

  // find calibration file 
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
