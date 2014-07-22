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
 *  synchronized within 1/4 of the cyle length (91 seconds). It is safer if we  
 *  assume a clock seconds difference that is less than 90, say 85 seconds. 
 *
 *  Then the algorithm for comparing A and B is:
 *
 *  First check if the clocks are more than 85 seconds apart. If so, use the 
 *  clock seconds alone to determine if A > B.
 *  If they are within 85 seconds, then also look at the fiducials.
 *
 *  Count the number of fiducials it takes to get from B to A.  Wrap around if 
 *  need be, that is if B is 0x1FFD0 and A is 0x10, then it takes 0x20 fiducials
 *  to get from B to A. Since the A and B dgrams have clock times are within 1/4 cycle, 
 *  and the max difference between between the A and B stream clocks is < 1/4 cycle, than the 
 *  difference between the corrected A and B clock times is less than 1/2 cycle
 *  (by corrected, we mean if both streams were exactly synchronized and
 *  on the same clock). The fiducials are synchronized, so if the number of fiducials to go 
 *  from B to A is greater than 1/2 the cycle, we are going the wrong way. A is before B.  
 *  However if going from B to A is < 1/2 the cycle, B is before A.
 *
 *  For A and B that are within 85 seconds of one another, it is possible
 *  to use the fiducials to compute the actual difference in the stream clocks, 
 *  to see how far apart they are. This is a rough computation as the fiducials 
 *  do not occur at exactly 1/360 seconds apart from one another.
 *
 *  It is important that the clocks are within 1/4 the fiducial cycle of one another.
 *  The comparision must be 'well ordered' meaning if A > B and B > C we must have A > C.
 *  This will not always be true if the clocks are more than 1/4 the fiducial cycle apart.
 *
 *  @version $Id$
 *
 *  @author David Schneider
 */

class FiducialsCompare {
public:
  /**
   *  @brief construct object to compare based on seconds/fiducials.
   *
   *  @param[in]  maxClockDriftSeconds - needs to be < 1/4 the fiducial cycle. Defaults to 85. 
   *              This is the maximum difference expected between the clocks (in seconds) when
   *              two elements are compared.
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

  /**
   *  @brief calculates difference in clocks based on difference in fiducials.
   *
   * When the A and B clock times are within 1/4 the fiducial cycle of one another,
   * one can look at how far apart the fiducials are to calculate the actual difference
   * in the clocks. Carries out this calculation using seconds and fiducials (not nanoseconds). 
   * Returns signed result in seconds: A-B. Result is rough, fiducials, while running at 
   * 360 Hz, are not precise to the nanosecond.
   */
  double fidBasedBtoASecondsDiff(const Pds::Dgram& A, const Pds::Dgram& B) const;

  /**
   *  @brief calculates difference in clocks based on difference in fiducials.
   *
   * When secondsA and secondsB are within 1/4 the fiducial cycle of one another,
   * one can look at how far apart the fiducials are to calculate the actual difference
   * in the clocks.  Returns signed result in seconds: A-B. Result is rough, fiducials, while 
   * running at 360 Hz, are not precise to the nanosecond.
   */
  double fidBasedBtoASecondsDiff(const uint32_t secondsA, const unsigned fiducialsA, 
                      const uint32_t secondsB, const unsigned fiducialsB) const;

  unsigned maxClockDriftSeconds() const { return m_maxClockDriftSeconds; }
private:
  unsigned m_maxClockDriftSeconds;
};

} // namespace XtcInput

#endif
