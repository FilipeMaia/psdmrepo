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
#include "psana/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------
namespace {
  const char * logger = "FiducialsCompare";

  const unsigned MaxFiducials = Pds::TimeStamp::MaxFiducials;
  const unsigned HalfFiducials = Pds::TimeStamp::MaxFiducials/2;
  const unsigned QuarterFiducials = HalfFiducials/2;

  /// how many fiducials does it take to get from A to B?
  /// take into account wrap around.
  unsigned fidDistanceWrap(unsigned fidA, unsigned fidB) {
    if (fidA >= MaxFiducials) MsgLog(logger, warning, "comparing fiducial value that is greater than max. "
                                     << " value=" << fidA << " max: " << MaxFiducials);
    if (fidB >= MaxFiducials) MsgLog(logger, warning, "comparing fiducial value that is greater than max. "
                                     << " value=" << fidB << " max: " << MaxFiducials);
    unsigned res;
    if (fidB >= fidA) {
      res = fidB - fidA;
    } else {
      res = fidB + (MaxFiducials-fidA);
    }
    return res;
  }

};

namespace XtcInput {

FiducialsCompare::FiducialsCompare(unsigned maxClockDriftSeconds) : 
  m_maxClockDriftSeconds(maxClockDriftSeconds) 
{
  if (maxClockDriftSeconds >= QuarterFiducials) {
    MsgLog(logger, 
           error, "FiducialsCompare initialized with "
           << maxClockDriftSeconds << " which is >= than 1/4 fiducial cycle");
  }
}
  
bool FiducialsCompare::fiducialsGreater(const Pds::Dgram& A, const Pds::Dgram& B) const
{
  return fiducialsGreater(A.seq.clock().seconds(), A.seq.stamp().fiducials(),
                          B.seq.clock().seconds(), B.seq.stamp().fiducials());
}

bool FiducialsCompare::fiducialsGreater(const uint32_t secondsA, const unsigned fiducialsA, 
                                        const uint32_t secondsB, const unsigned fiducialsB) const
{
  if (secondsA >= (secondsB + maxClockDriftSeconds())) return true;
  if ((secondsA + maxClockDriftSeconds()) <= secondsB) return false;

  // the clocks are within maxClockDriftSeconds of one another
  if (fiducialsA == fiducialsB) return false;
  
  // assuming maxClockDriftSeconds is < 1/4 Fiducial cycle, it is less than 
  // 1/2 way around the fiducials to get from one to the other.
  // If B to A is shorter, than B occurs before A in clock time, and A is greater.
  unsigned fidBtoA = fidDistanceWrap(fiducialsB, fiducialsA);
  bool AisGreater = (fidBtoA < HalfFiducials);
  return AisGreater;
}

double FiducialsCompare::fidBasedBtoASecondsDiff(const Pds::Dgram &A, const Pds::Dgram &B) const
{
  return fidBasedBtoASecondsDiff(A.seq.clock().seconds(), A.seq.stamp().fiducials(),
                                 B.seq.clock().seconds(), B.seq.stamp().fiducials());
}

double FiducialsCompare::fidBasedBtoASecondsDiff(const uint32_t secondsA, const unsigned fiducialsA, 
                                                   const uint32_t secondsB, const unsigned fiducialsB) const
{
  if ( (secondsA >= (secondsB + maxClockDriftSeconds())) or 
       ((secondsA + maxClockDriftSeconds()) <= secondsB) ) {
    throw psana::Exception(ERR_LOC,"fidBasedSecondsDiff called when clocks are to far apart");
  }

  unsigned fidBtoA = fidDistanceWrap(fiducialsB, fiducialsA);
  bool AisGreater = (fidBtoA < HalfFiducials);

  double BtoAdriftSec, fidBtoAsec;
  if (AisGreater) {
    fidBtoAsec = fidBtoA/360.0;
  } else {
    fidBtoAsec = -((MaxFiducials - fidBtoA)/360.0);
  }
  // unsigned arthimetic, get small integer before converting to double
  int BtoAsec;
  if (secondsA > secondsB) BtoAsec = secondsA-secondsB;
  else BtoAsec = -(secondsB-secondsA);

  BtoAdriftSec = double(BtoAsec) - fidBtoAsec;
  return BtoAdriftSec;
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
  if (secondsA >= secondsB) {
    if ((secondsA - secondsB) < maxClockDriftSeconds()) return true;
  } else {
    if ((secondsB - secondsA) < maxClockDriftSeconds()) return true;
  }
  return false;
}

} // namespace XtcInput
