//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: PnccdPedestalsV1.cpp 2014-01-24 11:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class PnccdPedestalsV1...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "pdscalibdata/PnccdPedestalsV1.h"
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
PnccdPedestalsV1::PnccdPedestalsV1() 
: PnccdBaseV1 ()
{
  std::fill_n(m_pars, int(Size), pars_t(0)); // All pixels have zero pedestal by default
}


PnccdPedestalsV1::PnccdPedestalsV1 (const std::string& fname) 
: PnccdBaseV1 ()
{
  load_pars_from_file <pars_t> (fname, "pedestals", Size, m_pars); 
}


void PnccdPedestalsV1::print()
{
  MsgLog("PnccdPedestalsV1", info, "pedestals:\n" << pedestals());
}

} // namespace pdscalibdata
