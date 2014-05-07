#ifndef PDSCALIBDATA_CSPADPIXELRMSV2_H
#define PDSCALIBDATA_CSPADPIXELRMSV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPixelRmsV2.
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

class CsPadPixelRmsV2: public CsPadBaseV2 {
public:

  typedef float pars_t;

  /// Default constructor
  CsPadPixelRmsV2()
  : CsPadBaseV2 ()
  {
    std::fill_n(m_pars, int(Size), pars_t(1)); // All pixels have unit rms by default
  }
  
  /// Read parameters from file
  CsPadPixelRmsV2 (const std::string& fname)
  : CsPadBaseV2 ()
  {
    load_pars_from_file <pars_t> (fname, "pixel_rms", Size, m_pars);
  }

  /// Destructor
  ~CsPadPixelRmsV2 (){}

  /// Access parameters
  ndarray<pars_t, 4> pixel_rms() const {
    return make_ndarray(m_pars, Quads, Segs, Rows, Cols);
  }

  /// Print array of parameters
  void  print()
  {
    MsgLog("CsPadPixelRmsV2", info, "pixel_rms:\n" << pixel_rms());
  }

protected:

private:

  /// Data members  
  mutable pars_t m_pars[Size];

  //std::string m_comment; 
  
  /// Copy constructor and assignment are disabled by default
  CsPadPixelRmsV2 ( const CsPadPixelRmsV2& ) ;
  CsPadPixelRmsV2& operator = ( const CsPadPixelRmsV2& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADPIXELRMSV2_H
