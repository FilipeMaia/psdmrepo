#ifndef PDSCALIBDATA_CSPADPIXELGAINV1_H
#define PDSCALIBDATA_CSPADPIXELGAINV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPixelGainV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <boost/utility.hpp>

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

class CsPadPixelGainV1 : boost::noncopyable {
public:

  enum { Quads = Pds::CsPad::MaxQuadsPerSensor };
  enum { Sections = Pds::CsPad::ASICsPerQuad/2 };
  enum { Columns = Pds::CsPad::ColumnsPerASIC };
  enum { Rows = Pds::CsPad::MaxRowsPerASIC*2 };
  enum { Size = Quads*Sections*Columns*Rows };
  
  typedef float pixelGain_t;
  
  // Default constructor
  CsPadPixelGainV1 () ;
  
  // read pixel gain from file
  CsPadPixelGainV1 (const std::string& fname) ;

  // Destructor
  ~CsPadPixelGainV1 () ;

  // access pixel gain data
  ndarray<pixelGain_t, 4> pixelGains() const {
    return make_ndarray(m_pixelGain, Quads, Sections, Columns, Rows);
  }

protected:

private:

  // Data members  
  mutable pixelGain_t m_pixelGain[Size];

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADPIXELGAINV1_H
