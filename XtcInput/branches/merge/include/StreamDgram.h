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

class StreamDgram : public Dgram {
 public:
  typedef enum {DAQ, controlUnderDAQ, controlIndependent} StreamType;  

  // streamId is for clients to keep track of StreamDgram's that go in and out of
  // a priority queue. It is not used for comparisons.
 StreamDgram(const Dgram &dgram, StreamType streamType, int64_t L1block, int streamId) 
   : Dgram(dgram), m_streamType(streamType), m_L1block(L1block), m_streamId(streamId)
    {}

  /**
   *  Default ctor
   */
  StreamDgram() : Dgram(), m_streamType(DAQ), m_L1block(-1) {}

  /**
   *  If this dgram is an L1Accept, L1Block returns a 0-up counter that is the
   *  block of L1Accepts that this Dgram is a part of.  If this dgram is a 
   *  Transition, returns the number of L1Accept blocks that precede it.
   *  For example:
   *  dgram stream: T T L L L L L L T T L L L L T T L L T T L L L T L L T  
   *  L1Block:      0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4
   *
   *  The counter must increment for 'empty' L1Blocks, that can occur when a enable
   *  Transition is followed a disable transtion. 
   *
   *  The L1Block is relative to a run. When the run changes, the L1Block should be
   *  reset to 0.
   *
   *  Within a run, the L1Block counts the number of disable transitions up to and 
   *  including the current transition.
   *
   *  It is important to construct StreamDgram with the correct L1Block value so that
   *  StreamDgramCmp will function correctly.
   *
   *  If this dgram is empty, returns, -1.
   *
   *  From the above example, one sees that within a run, 
   *
   *           T > L     if      L1Block(T) > L1Block(L) 
   *           L > T     if      L1Block(L) >= L1Block(T) 
  */
  int64_t L1Block() const { 
    if (empty()) return -1; 
    return m_L1block; 
  };

  StreamType streamType() const { return m_streamType; }

  int streamId() const { return m_streamId; }

  static std::string streamType2str(const StreamType type); // get string rep
  static std::string dumpStr(const StreamDgram &dg);  // dump object to string, for debugging

 private:

  StreamType m_streamType;
  int64_t m_L1block;
  int m_streamId;
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
                  unsigned maxClockDrift = 90);

  /// operator greater than for two StreamDgram's
  bool operator()(const StreamDgram &a, const StreamDgram &b) const;

  /// determine if two StreamDgram's are part of the same event
  bool sameEvent(const StreamDgram &a, const StreamDgram &b) const;

 protected:
  // types for determining the category of a StreamDgram, and a pair of StreamDgram's
  typedef enum {L1Accept, otherTrans} TransitionType;
  typedef std::pair<TransitionType, StreamDgram::StreamType> DgramCategory;
  typedef std::pair<DgramCategory, DgramCategory> DgramCategoryAB;
  static DgramCategory getDgramCategory(const StreamDgram &dg);
  static DgramCategoryAB makeDgramCategoryAB(DgramCategory a, DgramCategory b);

  /// the different kinds of comparisons
  typedef enum {clockCmp, fidCmp, blockCmp, mapCmp, badCmp} CompareMethod;  
  bool doClockCmp(const StreamDgram &a, const StreamDgram &b) const;
  bool doFidCmp(const StreamDgram &a, const StreamDgram &b) const;
  bool doBlockCmp(const StreamDgram &a, const StreamDgram &b) const;
  bool doMapCmp(const StreamDgram &a, const StreamDgram &b) const;
  bool doBadCmp(const StreamDgram &a, const StreamDgram &b) const;

  // helper function for above. Based on run #, returns
  // a < b: -1,   a==b 0   a > b: 1
  int runLessGreater(const StreamDgram &a, const StreamDgram &b) const;

  // helper function for above. Based on block #, returns
  // a < b: -1,   a==b 0   a > b: 1
  int blockLessGreater(const StreamDgram &a, const StreamDgram &b) const;
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
