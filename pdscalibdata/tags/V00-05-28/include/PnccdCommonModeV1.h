#ifndef PDSCALIBDATA_PNCCDCOMMONMODEV1_H
#define PDSCALIBDATA_PNCCDCOMMONMODEV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnccdCommonModeV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------
#include "pdscalibdata/PnccdBaseV1.h" // Segs, Rows, Cols etc.
#include "pdscalibdata/GlobalMethods.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ndarray/ndarray.h"
#include "pdsdata/psddl/pnccd.ddl.h"

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

class PnccdCommonModeV1: public PnccdBaseV1 {
public:

  const static size_t CMSize = 16; 
 
  typedef double pars_t;

  /// Default constructor
  PnccdCommonModeV1 ()
  : PnccdBaseV1 ()
  {
    std::fill_n(m_pars, int(CMSize), pars_t(0));
    // default common_mode parameters
    m_pars[0] = 2;   // Common mode algorithm number
    m_pars[1] = 300; // Threshold; values below threshold are averaged
    m_pars[2] = 50;  // Maximal correction; correction is applied if it is less than this value
    m_pars[3] = 128; // Number of pixels for averaging
    m_pars[4] = 0;   // Fraction of pixels in cm peak ...
  }

  /// Read parameters from file
  PnccdCommonModeV1 (const std::string& fname)
  : PnccdBaseV1 ()
  {
    std::fill_n(m_pars, int(CMSize), pars_t(0));
    load_pars_from_file <pars_t> (fname, "common_mode", CMSize, m_pars, 2);
  }

  /// Destructor
  ~PnccdCommonModeV1 (){}

  /// Access parameters
  ndarray<pars_t, 1> common_mode() const {
    return make_ndarray(m_pars, CMSize);
  }

  /// Print array of parameters
  void  print()
  {
    MsgLog("PnccdCommonModeV1", info, "common_mode:\n" << common_mode());
  }

protected:

private:

  /// Data members  
  mutable pars_t m_pars[CMSize];

  //std::string m_comment; 
  
  /// Copy constructor and assignment are disabled by default
  PnccdCommonModeV1 ( const PnccdCommonModeV1& ) ;
  PnccdCommonModeV1& operator = ( const PnccdCommonModeV1& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_PNCCDCOMMONMODEV1_H
