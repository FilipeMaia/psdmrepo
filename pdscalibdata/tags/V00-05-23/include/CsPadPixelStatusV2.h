#ifndef PDSCALIBDATA_CSPADPIXELSTATUSV2_H
#define PDSCALIBDATA_CSPADPIXELSTATUSV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPixelStatusV2.
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
//#include "pdsdata/psddl/cspad.ddl.h"

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

class CsPadPixelStatusV2: public CsPadBaseV2 {
public:

  typedef uint16_t pars_t;

  /// Default constructor
  CsPadPixelStatusV2()
  : CsPadBaseV2 ()
  {
    std::fill_n(m_pars, int(Size), pars_t(0)); // All pixels have good 0-status by default
  }
  
  /// Read parameters from file
  CsPadPixelStatusV2 (const std::string& fname)
  : CsPadBaseV2 ()
  {
    load_pars_from_file <pars_t> (fname, "pixel_status", Size, m_pars);
  }

  /// Destructor
  ~CsPadPixelStatusV2 (){}

  /// Access parameters
  ndarray<pars_t, 4> pixel_status() const {
    return make_ndarray(m_pars, Quads, Segs, Rows, Cols);
  }

  /// Print array of parameters
  void  print()
  {
    MsgLog("CsPadPixelStatusV2", info, "pixel_status:\n" << pixel_status());
  }

protected:

private:

  /// Data members  
  mutable pars_t m_pars[Size];

  //std::string m_comment; 
  
  /// Copy constructor and assignment are disabled by default
  CsPadPixelStatusV2 ( const CsPadPixelStatusV2& ) ;
  CsPadPixelStatusV2& operator = ( const CsPadPixelStatusV2& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADPIXELSTATUSV2_H
