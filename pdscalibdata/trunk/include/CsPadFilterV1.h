#ifndef PDSCALIBDATA_CSPADFILTERV1_H
#define PDSCALIBDATA_CSPADFILTERV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadFilterV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>
#include <stdint.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/**
 *  CsPad calibration data class which is actually a filter.
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

class CsPadFilterV1  {
public:

  enum FilterMode { 
    None = 0,
    FullImage = 1
  };
  
  enum { DataSize = 16 };

  // Default constructor, pass-throug
  CsPadFilterV1 ();

  /// Read constants from file
  CsPadFilterV1 (const std::string& fname) ;

  /// Initialize constants from parameters
  CsPadFilterV1 (FilterMode mode, const double data[DataSize]) ;

  // Destructor
  ~CsPadFilterV1 () ;

  // access filter mode
  FilterMode mode() const { return FilterMode(m_mode); }

  // access filter data
  const double* data() const { return m_data; }

  /**
   *  Returns yes/no decision for the given data.
   *   
   *  @param pixelData    Pixel data from cspad, after pedestal (and 
   *                      optionally common mode) subtraction
   */
  bool filter(const ndarray<const int16_t, 3>& pixelData) const;
  
  /**
   *  Returns yes/no decision for the given data.
   *
   *  @param pixelData    Pixel data from cspad, after pedestal (and
   *                      optionally common mode) subtraction
   *  @param pixelStatus  Pixel status data
   */
  bool filter(const ndarray<const int16_t, 3>& pixelData, const ndarray<const uint16_t, 3>& pixelStatus) const;

protected:

private:

  // Data members
  // Data members  
  uint32_t m_mode;
  double m_data[DataSize];

};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADFILTERV1_H
