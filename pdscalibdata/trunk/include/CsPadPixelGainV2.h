#ifndef PDSCALIBDATA_CSPADPIXELGAINV2_H
#define PDSCALIBDATA_CSPADPIXELGAINV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPixelGainV2.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------
#include "pdscalibdata/CsPadBaseV2.h" // Quads, Segs, Rows, Cols etc.
#include "pdscalibdata/GlobalMethods.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
// #include "pdsdata/psddl/cspad.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "MsgLogger/MsgLogger.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pdscalibdata {

/**
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

class CsPadPixelGainV2: public CsPadBaseV2 {
public:

  typedef float pars_t;

  /// Default constructor
  CsPadPixelGainV2()
  : CsPadBaseV2 ()
  {
    std::fill_n(m_pars, int(Size), pars_t(1)); // All pixels have unit rms by default
  }
  
  /// Read parameters from file
  CsPadPixelGainV2 (const std::string& fname)
  : CsPadBaseV2 ()
  {
    load_pars_from_file <pars_t> (fname, "pixel_gain", Size, m_pars);
  }

  /// Destructor
  ~CsPadPixelGainV2 (){}

  /// Access parameters
  ndarray<pars_t, 4> pixel_gain() const {
    return make_ndarray(m_pars, Quads, Segs, Rows, Cols);
  }

  /// Print array of parameters
  void  print()
  {
    MsgLog("CsPadPixelGainV2", info, "pixel_gain:\n" << pixel_gain());
  }

protected:

private:

  /// Data members  
  mutable pars_t m_pars[Size];

  //std::string m_comment; 
  
  /// Copy constructor and assignment are disabled by default
  CsPadPixelGainV2 ( const CsPadPixelGainV2& ) ;
  CsPadPixelGainV2& operator = ( const CsPadPixelGainV2& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADPIXELGAINV2_H
