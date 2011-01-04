//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeConstants...
//      contains constants for the Time and Duration classes.
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "PSTime/TimeConstants.h"

namespace PSTime {

//----------------
// Definitions  --
//----------------

const time_t TimeConstants::s_minusInfinity = 0U;           // 0
const time_t TimeConstants::s_plusInfinity  = 4294967295U;  // 2^32 - 1
const time_t TimeConstants::s_nsecInASec    = 1000000000U;  // ns in 1 sec 

} // namespace PSTime
