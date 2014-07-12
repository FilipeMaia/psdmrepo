#ifndef FIDUCIALSCOMPARE_H
#define FIDUCIALSCOMPARE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FidcualsCompare.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <stdint.h>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-------------------------------
// Collaborating Class Declarations  --
//-------------------------------
namespace Pds {
  class Dgram;
};

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief for sorting datagrams by fiducials and rough match on seconds
 *
 *  Fiducials go through the values [0,0x1FFE0) at 360HZ. They wrap at 364 seconds.
 *  Suppose A and B are the clock/fiducials for two events where the clocks are 
 *  synchronized within 1/4 of the cyle length (91 seconds). I'm pretty sure this is
 *  maximum difference we can allow, but not positive. To be safe, since we do not 
 *  use nanoseconds, use something less than 90, say 85 seconds. 
 *
 *  Then the algorithm for comparing A and B is:
 *
 *  First check if the clocks using the maximum difference between them.
 *  If they are within the maximum difference, look at the fiducials.
 *
 *  Count the number of fiducials it takes to get from B to A.  Wrap around if 
 *  need be, that is if B is 0x1FFD0 and A is 0x10, then it takes 0x20 fiducials
 *  to get from B to A. Since A and B clock times are within 1/4 cycle, and the max
 *  difference between between the A and B stream clocks is < 1/4, the difference in 
 *  clock times between A and B is less than 1/2 cycle. If the number of fiducials to 
 *  go from B to A is greater than 1/2 the cycle, we are going the wrong way. 
 *  A is before B.  However if going from B to A is < 1/2 the cycle, B is before A.
 *
 *  After computing the fiducial distance and wether or not A > B (this can all
 *  be done with integer arthimetic) one can compute  the a actual difference in the
 *  clocks. Presently, if we see a large difference, we report this, but in the
 *  future we may not want to waste time on the computation.
 *
 *  @version $Id$
 *
 *  @author David Schneider
 */

class FiducialsCompare {
public:
  /**
   *  @brief provides functions to compare fiducials.
   *
   *  @param[in]  maxClockDriftSeconds fiducials equal plus seconds within this value are the same event
   */
  FiducialsCompare(unsigned maxClockDriftSeconds = 85);

  // return true is A > B
  bool fiducialsGreater(const Pds::Dgram& A, const Pds::Dgram& B) const;

  // return true is A > B
  bool fiducialsGreater(const uint32_t secondsA, const unsigned fiducialsA, 
                        const uint32_t secondsB, const unsigned fiducialsB) const;

  bool fiducialsEqual(const Pds::Dgram& A, const Pds::Dgram& B) const;

  bool fiducialsEqual(const uint32_t secondsA, const unsigned fiducialsA, 
                      const uint32_t secondsB, const unsigned fiducialsB) const;

  unsigned maxClockDriftSeconds() const { return m_maxClockDriftSeconds; }
private:
  unsigned m_maxClockDriftSeconds;
};

} // namespace XtcInput

#endif
