//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: PnccdPixelRmsV1.cpp 2014-02-08 18:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class PnccdPixelRmsV1...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/PnccdPixelRmsV1.h"
#include "pdscalibdata/GlobalMethods.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <stdexcept>
#include <fstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pdscalibdata {

//----------------
// Constructors --
//----------------
PnccdPixelRmsV1::PnccdPixelRmsV1() 
: PnccdBaseV1 ()
{
  std::fill_n(m_pars, int(Size), pars_t(1)); // All pixels have unit rms by default
  // std::fill_n(m_pars, int(Size), pars_t(0.5)); // For test purpose only!
}


PnccdPixelRmsV1::PnccdPixelRmsV1 (const std::string& fname) 
: PnccdBaseV1 ()
{
  load_pars_from_file <pars_t> (fname, "pixel_rms", Size, m_pars); 
}


void PnccdPixelRmsV1::print()
{
  MsgLog("PnccdPixelRmsV1", info, "pixel_rms:\n" << pixel_rms());
}

} // namespace pdscalibdata
