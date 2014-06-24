#ifndef XTCINPUT_STREAMDGRAM_H
#define XTCINPUT_STREAMDGRAM_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: Dgram.h 8209 2014-05-14 02:49:47Z cpo@SLAC.STANFORD.EDU $
//
// Description:
//	Class Dgram.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include "boost/shared_ptr.hpp"
//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psana/Exceptions.h"
#include "XtcInput/Dgram.h"
#include "XtcInput/FiducialsCompare.h"
//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

typedef enum {DAQ, controlUnderDAQ, controlIndependent} StreamType;  

class StreamDgram : public Dgram {
 public:

 StreamDgram(const Dgram &dgram, StreamType streamType, int64_t L1block, int streamIndex) 
   : Dgram(dgram), m_streamType(streamType), m_L1block(L1block), m_streamIndex(streamIndex)
    {}

  /**
   *  Default ctor
   */
  StreamDgram() : Dgram(), m_streamType(DAQ), m_L1block(-1) {}

  
  /// The L1Block is relative to the run within a stream. Code that constructs StreamDgram
  /// must guarentee this, so that code that uses StreamDgram can rely on this.
  /// If this dgram is an L1Accept, returns the block of L1Accepts that
  /// this datagram is a part of in its run of the stream. If this dgram is a Transition,
  /// returns the number of L1Accept blocks that preceded it in the stream.
  /// if this dgram is empty, returns, -1.
  int64_t L1Block() const { 
    if (empty()) return -1; 
    return m_L1block; 
  };

  StreamType streamType() const { return m_streamType; }

  int streamIndex() const { return m_streamIndex; }

 private:

  StreamType m_streamType;
  int64_t m_L1block;
  int m_streamIndex;
};

/// Implements operator() which returns true if a > b, (Greater than) for use
/// with a priority queue where we want the first element is the one with the
/// least time (as opposed to highest)
class StreamDgramCmp {
 public:
  /// typedefs for defining clock drift between different experiments
  typedef std::pair<unsigned, unsigned> ExperimentPair;
  typedef std::pair<uint32_t, uint32_t> ClockDiff;
  typedef std::map<ExperimentPair, ClockDiff> ExperimentClockDiffMap;

  /// constructor, optional clock drift map can be passed in
  StreamDgramCmp(const boost::shared_ptr<ExperimentClockDiffMap> expClockDiff = 
                  boost::shared_ptr<ExperimentClockDiffMap>(),
                  unsigned maxClockDrift = 120);

  /// operator greater than for two StreamDgram's
  bool operator()(const StreamDgram &a, const StreamDgram &b) const;

  /// determine if two StreamDgram's are part of the same event
  bool sameEvent(const StreamDgram &a, const StreamDgram &b) const;

 protected:
  // types for determining the category of a StreamDgram, and a pair of StreamDgram's
  typedef enum {L1Accept, otherTrans} TransitionType;
  typedef std::pair<TransitionType, StreamType> DgramCategory;
  typedef std::pair<DgramCategory, DgramCategory> DgramCategoryAB;
  static DgramCategory getDgramCategory(const StreamDgram &dg);
  static DgramCategoryAB makeDgramCategoryAB(DgramCategory a, DgramCategory b);

  /// the different kinds of comparisons
  typedef enum {clockCmp, fidCmp, blockCmp, mapCmp} CompareMethod;  
  bool doClockCmp(const StreamDgram &a, const StreamDgram &b) const;
  bool doFidCmp(const StreamDgram &a, const StreamDgram &b) const;
  bool doBlockCmp(const StreamDgram &a, const StreamDgram &b) const;
  bool doMapCmp(const StreamDgram &a, const StreamDgram &b) const;
 private:
  const boost::shared_ptr<ExperimentClockDiffMap> m_expClockDiff;
  std::map<DgramCategoryAB, CompareMethod> m_LUT;
  FiducialsCompare m_fidCompare;

 public:
 // UnknownCmp is thrown when two dgrams are compared for which no comparision
 // method is known - most likely an internal logic error in StreamDgramCmp
 class UnknownCmp : public psana::Exception {
 public:
 UnknownCmp(const ErrSvc::Context &ctx) : 
   psana::Exception(ctx, "unknown cmp method - StreamDgramCmp") {}
 };

 // NoClockDiff is thrown when a mapCmp is requries, but either the experiments for the
 // streams compared was not valid, or no clockDiff was specified for the two experiments
 class NoClockDiff : public psana::Exception {
 public:
 NoClockDiff(const ErrSvc::Context &ctx, unsigned expA, unsigned expB) :
   psana::Exception(ctx, "no clock diff for experiments")
     , m_expA(expA)
     , m_expB(expB) {};
   unsigned m_expA, m_expB;
 };
};

} // namespace XtcInput

#endif // XTCINPUT_DGRAM_H
