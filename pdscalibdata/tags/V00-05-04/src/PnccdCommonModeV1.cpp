//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: PnccdCommonModeV1.cpp 2014-01-24 11:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class PnccdCommonModeV1...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/PnccdCommonModeV1.h"
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
PnccdCommonModeV1::PnccdCommonModeV1() 
: PnccdBaseV1 ()
{
  //std::fill_n(m_pars, int(CMSize), pars_t(0));
  m_pars[0] = 1; //= {1, 50, 10}; - default common_mode parameters
  m_pars[1] = 50;
  m_pars[2] = 10;
}


PnccdCommonModeV1::PnccdCommonModeV1 (const std::string& fname) 
: PnccdBaseV1 ()
{
  load_pars_from_file <pars_t> (fname, "common_mode", CMSize, m_pars); 
}


void PnccdCommonModeV1::print()
{
  MsgLog("PnccdCommonModeV1", info, "common_mode:\n" << common_mode());
}

} // namespace pdscalibdata
