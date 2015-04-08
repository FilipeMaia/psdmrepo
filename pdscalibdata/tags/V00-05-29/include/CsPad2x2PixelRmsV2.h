#ifndef PDSCALIBDATA_CSPAD2X2PIXELRMSV2_H
#define PDSCALIBDATA_CSPAD2X2PIXELRMSV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2PixelRmsV2.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------
#include "pdscalibdata/CsPad2x2BaseV2.h" // Segs, Rows, Cols etc.
#include "pdscalibdata/GlobalMethods.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
// #include "pdsdata/psddl/cspad2x2.ddl.h"

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

class CsPad2x2PixelRmsV2: public CsPad2x2BaseV2 {
public:

  typedef float pars_t;

  /// Default constructor
  CsPad2x2PixelRmsV2()
  : CsPad2x2BaseV2 ()
  {
    std::fill_n(m_pars, int(Size), pars_t(1)); // All pixels have unit rms by default
  }
  
  /// Read parameters from file
  CsPad2x2PixelRmsV2 (const std::string& fname)
  : CsPad2x2BaseV2 ()
  {
    load_pars_from_file <pars_t> (fname, "pixel_rms", Size, m_pars);
  }

  /// Destructor
  ~CsPad2x2PixelRmsV2 (){}

  /// Access parameters
  ndarray<pars_t, 3> pixel_rms() const {
    return make_ndarray(m_pars, Rows, Cols, Segs);
  }

  /// Print array of parameters
  void print()
  {
    MsgLog("CsPad2x2PixelRmsV2", info, "pixel_rms:\n" << pixel_rms());
  }

protected:

private:

  /// Data members  
  mutable pars_t m_pars[Size];

  //std::string m_comment; 
  
  /// Copy constructor and assignment are disabled by default
  CsPad2x2PixelRmsV2 ( const CsPad2x2PixelRmsV2& ) ;
  CsPad2x2PixelRmsV2& operator = ( const CsPad2x2PixelRmsV2& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPAD2X2PIXELRMSV2_H
