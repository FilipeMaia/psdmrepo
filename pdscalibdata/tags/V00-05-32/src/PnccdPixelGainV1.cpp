//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: PnccdPixelGainV1.cpp 2014-01-24 11:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class PnccdPixelGainV1...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/PnccdPixelGainV1.h"
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
PnccdPixelGainV1::PnccdPixelGainV1() 
: PnccdBaseV1 ()
{
  std::fill_n(m_pars, int(Size), pars_t(1)); // All pixels have unit gain by default
  // std::fill_n(m_pars, int(Size), pars_t(0.5)); // For test purpose only!
}


PnccdPixelGainV1::PnccdPixelGainV1 (const std::string& fname) 
: PnccdBaseV1 ()
{
  load_pars_from_file <pars_t> (fname, "pixel_gain", Size, m_pars); 
}


void PnccdPixelGainV1::print()
{
  MsgLog("PnccdPixelGainV1", info, "pixel_gain:\n" << pixel_gain());
}

} // namespace pdscalibdata
