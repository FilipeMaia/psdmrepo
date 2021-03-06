#ifndef PDSCALIBDATA_CSPADPIXELSTATUSV1_H
#define PDSCALIBDATA_CSPADPIXELSTATUSV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPixelStatusV1.
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
#include "pdsdata/cspad/Detector.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/**
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class CsPadPixelStatusV1  {
public:

  enum { Quads = Pds::CsPad::MaxQuadsPerSensor };
  enum { Sections = Pds::CsPad::ASICsPerQuad/2 };
  enum { Columns = Pds::CsPad::ColumnsPerASIC };
  enum { Rows = Pds::CsPad::MaxRowsPerASIC*2 };
  enum { Size = Quads*Sections*Columns*Rows };
  
  // This codes must be the same as in CspadCorrector
  enum PixelStatus { 
    VeryHot=1,
    Hot=2,
    Cold=4
  };
  
  typedef uint16_t status_t;
  typedef status_t StatusCodes[Quads][Sections][Columns][Rows];
  
  /// Default constructor, all pixel codes set to 0
  CsPadPixelStatusV1 () ;
  
  /**
   *  Read all codes from file.
   *  
   *  @throw std::exception
   */
  CsPadPixelStatusV1 (const std::string& fname) ;

  // Destructor
  ~CsPadPixelStatusV1 () ;

  // access status data
  const StatusCodes& status() const { return m_status; }

protected:

private:

  // Data members  
  StatusCodes m_status;

  // Copy constructor and assignment are disabled by default
  CsPadPixelStatusV1 ( const CsPadPixelStatusV1& ) ;
  CsPadPixelStatusV1& operator = ( const CsPadPixelStatusV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADPIXELSTATUSV1_H
