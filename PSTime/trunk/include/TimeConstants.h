#ifndef PSTIME_TIMECONSTANTS_H
#define PSTIME_TIMECONSTANTS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeConstants
//      contains constants for the Time and Duration classes.
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <time.h>

namespace PSTime {

class TimeConstants  {
public:

  static const time_t s_minusInfinity;   // -infinity = 0
  static const time_t s_plusInfinity;    // +infinity = 2^32 - 1
  static const time_t s_nsecInASec;      // # of nanoseconds in one second

};

} // namespace PSTime

#endif // PSTIME_TIMECONSTANTS_H
