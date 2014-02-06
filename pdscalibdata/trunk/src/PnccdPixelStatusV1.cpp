//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: PnccdPixelStatusV1.cpp 2014-01-24 11:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class PnccdPixelStatusV1...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/PnccdPixelStatusV1.h"
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
PnccdPixelStatusV1::PnccdPixelStatusV1() 
: PnccdBaseV1 ()
{
  std::fill_n(m_pars, int(Size), pars_t(0)); // All pixels are good by default
  //std::fill_n(&m_pars[Size/4], int(Size/8), pars_t(1)); // For test purpose only add bad pixels in center!
}


PnccdPixelStatusV1::PnccdPixelStatusV1 (const std::string& fname) 
: PnccdBaseV1 ()
{
  load_pars_from_file <pars_t> (fname, "pixel_status", Size, m_pars); 
}


void PnccdPixelStatusV1::print()
{
  MsgLog("PnccdPixelStatusV1", info, "pixel_status:\n" << pixel_status());
}

} // namespace pdscalibdata
