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
 *  Fiducials go through the values [0,0x1FFE0) at 360HZ. They wrap at 364 seconds
 *  or 6 minutes and 4 seconds. Suppose A and B are the clock/fiducials for two
 *  events where the clocks are synchronized within 2 minutes of one another.
 *  Then the algorithm for comparing A and B is:
 *
 *  deltaFid = (B.fiducials - A.fiducials) mod 0x1FFE0
 *    where we mean the wrap distance, as a positive number, i.e, 
 *    (0x1FFD0 - 0x10) mod 0x1FFE0) == 0x20
 *  deltaFidSec = deltaFid/360.0
 *  earilestBclockIfBGreaterThanA = A.clock + deltaFidSec - 2 minutes
 *  if B.clock > earilestBclockIfBGreaterThanA then B > A
 * 
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
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
  FiducialsCompare(unsigned maxClockDriftSeconds = 90);

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
