#ifndef PDSCALIBDATA_CSPAD2X2COMMONMODEV2_H
#define PDSCALIBDATA_CSPAD2X2COMMONMODEV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2CommonModeV2.
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

class CsPad2x2CommonModeV2: public CsPad2x2BaseV2 {
public:

  const static size_t CMSize = 16; 
 
  typedef double pars_t;

  /// Default constructor
  CsPad2x2CommonModeV2 ()
  : CsPad2x2BaseV2 ()
  {
    std::fill_n(m_pars, int(CMSize), pars_t(0));
    // default common_mode parameters
    m_pars[0] = 1;   // Common mode algorithm number
    m_pars[1] = 25;  // Threshold; values below threshold are averaged
    m_pars[2] = 10;  // Maximal correction; correction is applied if it is less than this value
  }

  /// Read parameters from file
  CsPad2x2CommonModeV2 (const std::string& fname)
  : CsPad2x2BaseV2 ()
  {
    std::fill_n(m_pars, int(CMSize), pars_t(0));
    load_pars_from_file <pars_t> (fname, "common_mode", CMSize, m_pars, 2);
  }

  /// Destructor
  ~CsPad2x2CommonModeV2 (){}

  /// Access parameters
  ndarray<pars_t, 1> common_mode() const {
    return make_ndarray(m_pars, CMSize);
  }

  /// Print array of parameters
  void  print()
  {
    MsgLog("CsPad2x2CommonModeV2", info, "common_mode:\n" << common_mode());
  }

protected:

private:

  /// Data members  
  mutable pars_t m_pars[CMSize];

  //std::string m_comment; 
  
  /// Copy constructor and assignment are disabled by default
  CsPad2x2CommonModeV2 ( const CsPad2x2CommonModeV2& ) ;
  CsPad2x2CommonModeV2& operator = ( const CsPad2x2CommonModeV2& ) ;
};

} // namespace pdscalibdata

#endif // PDSCALIBDATA_CSPAD2X2COMMONMODEV2_H
