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
#include <stdint.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
#include "pdsdata/psddl/cspad.ddl.h"

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
  ndarray<status_t, 4> status() const {
    return make_ndarray(m_status, Quads, Sections, Columns, Rows);
  }

protected:

private:

  // Data members  
  mutable status_t m_status[Size];

  // Copy constructor and assignment are disabled by default
  CsPadPixelStatusV1 ( const CsPadPixelStatusV1& ) ;
  CsPadPixelStatusV1& operator = ( const CsPadPixelStatusV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADPIXELSTATUSV1_H
