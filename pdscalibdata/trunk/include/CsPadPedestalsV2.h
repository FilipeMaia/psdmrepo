#ifndef PDSCALIBDATA_CSPADPEDESTALSV2_H
#define PDSCALIBDATA_CSPADPEDESTALSV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadPedestalsV2.
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

class CsPadPedestalsV2: public CsPadBaseV2 {
public:

  typedef float pars_t;

  /// Default constructor
  CsPadPedestalsV2()
  : CsPadBaseV2 ()
  {
    std::fill_n(m_pars, int(Size), pars_t(0)); // All pixels have unit rms by default
  }
  
  /// Read parameters from file
  CsPadPedestalsV2 (const std::string& fname)
  : CsPadBaseV2 ()
  {
    load_pars_from_file <pars_t> (fname, "pedestals", Size, m_pars);
  }

  /// Destructor
  ~CsPadPedestalsV2 (){}

  /// Access parameters
  ndarray<pars_t, 4> pedestals() const {
    return make_ndarray(m_pars, Quads, Segs, Rows, Cols);
  }

  /// Print array of parameters
  void  print()
  {
    MsgLog("CsPadPedestalsV2", info, "pedestals:\n" << pedestals());
  }

protected:

private:

  /// Data members  
  mutable pars_t m_pars[Size];

  //std::string m_comment; 
  
  /// Copy constructor and assignment are disabled by default
  CsPadPedestalsV2 ( const CsPadPedestalsV2& ) ;
  CsPadPedestalsV2& operator = ( const CsPadPedestalsV2& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADPEDESTALSV2_H
