#ifndef PDSCALIBDATA_CSPADCOMMONMODEV2_H
#define PDSCALIBDATA_CSPADCOMMONMODEV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadCommonModeV2.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------
#include "pdscalibdata/CsPadBaseV2.h" // Segs, Rows, Cols etc.
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

class CsPadCommonModeV2: public CsPadBaseV2 {
public:

  const static size_t CMSize = 16; 
 
  typedef double pars_t;

  /// Default constructor
  CsPadCommonModeV2 ()
  : CsPadBaseV2 ()
  {
    std::fill_n(m_pars, int(CMSize), pars_t(0));
    // default common_mode parameters
    m_pars[0] = 1;   // Common mode algorithm number
    m_pars[1] = 25;  // Threshold; values below threshold are averaged
    m_pars[2] = 10;  // Maximal correction; correction is applied if it is less than this value
  }

  /// Read parameters from file
  CsPadCommonModeV2 (const std::string& fname)
  : CsPadBaseV2 ()
  {
    std::fill_n(m_pars, int(CMSize), pars_t(0));
    load_pars_from_file <pars_t> (fname, "common_mode", CMSize, m_pars, 2);
  }

  /// Destructor
  ~CsPadCommonModeV2 (){}

  /// Access parameters
  ndarray<pars_t, 1> common_mode() const {
    return make_ndarray(m_pars, CMSize);
  }

  /// Print array of parameters
  void  print()
  {
    MsgLog("CsPadCommonModeV2", info, "common_mode:\n" << common_mode());
  }

protected:

private:

  /// Data members  
  mutable pars_t m_pars[CMSize];

  //std::string m_comment; 
  
  /// Copy constructor and assignment are disabled by default
  CsPadCommonModeV2 ( const CsPadCommonModeV2& ) ;
  CsPadCommonModeV2& operator = ( const CsPadCommonModeV2& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPADCOMMONMODEV2_H
