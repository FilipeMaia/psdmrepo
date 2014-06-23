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
  FiducialsCompare(unsigned maxClockDriftSeconds = 120);

  bool fiducialsGreater(const Pds::Dgram& A, const Pds::Dgram& B) const;
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
