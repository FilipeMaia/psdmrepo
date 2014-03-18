//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FiducialsCompare
//
// Author List:
//      David Schneider
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/FiducialsCompare.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Dgram.hh"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------
namespace XtcInput {

FiducialsCompare::FiducialsCompare(unsigned maxClockDriftSeconds) : 
  m_maxClockDriftSeconds(maxClockDriftSeconds) 
{
  // TODO: error checking clock drift, make sure it is within fiducials cycle length
}
  
bool FiducialsCompare::fiducialsGreater(const Pds::Dgram& A, const Pds::Dgram& B) const
{
  return fiducialsGreater(A.seq.clock().seconds(), A.seq.stamp().fiducials(),
                          B.seq.clock().seconds(), B.seq.stamp().fiducials());
}

bool FiducialsCompare::fiducialsGreater(const uint32_t secondsA, const unsigned fiducialsA, 
                                        const uint32_t secondsB, const unsigned fiducialsB) const
{
  int64_t secondsDiff = int64_t(secondsA)-int64_t(secondsB);
  if (secondsDiff >  int64_t(m_maxClockDriftSeconds)) return true;
  if (secondsDiff < -int64_t(m_maxClockDriftSeconds)) return false;
  
  return fiducialsA > fiducialsB;
}

bool FiducialsCompare::fiducialsEqual(const Pds::Dgram& A, const Pds::Dgram& B) const
{
  return fiducialsEqual(A.seq.clock().seconds(), A.seq.stamp().fiducials(),
                        B.seq.clock().seconds(), B.seq.stamp().fiducials());
}

bool FiducialsCompare::fiducialsEqual(const uint32_t secondsA, const unsigned fiducialsA, 
                                      const uint32_t secondsB, const unsigned fiducialsB) const 
{
  if (fiducialsA != fiducialsB) return false;
  return std::abs(int64_t(secondsA) - int64_t(secondsB)) < m_maxClockDriftSeconds;
}

} // namespace XtcInput
