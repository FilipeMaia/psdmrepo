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

  StreamDgram(const Dgram &dgram, StreamType streamType, int64_t L1block) 
    : Dgram(dgram), m_streamType(streamType), m_L1block(L1block)
    {}

  /**
   *  Default ctor
   */
  StreamDgram() : Dgram(), m_streamType(DAQ), m_L1block(-1) {}

  
  /// if this dgram is an L1Accept, returns the block of L1Accepts that
  /// this datagram is a part of in the stream. If this dgram is a Transition,
  /// returns the number of L1Accept blocks that preceded it in the stream.
  /// if this dgram is empty, returns, -1.
  int64_t L1Block() const { 
    if (empty()) return -1; 
    return m_L1block; 
  };

  StreamType streamType() const { return m_streamType; }
private:

  StreamType m_streamType;
  int64_t m_L1block;
};

/// determines if this dgram should appear earlier in a merge with others
class LessStreamDgram {
 public:
  /// typedefs for defining clock drift between different experiments
  typedef std::pair<unsigned, unsigned> ExperimentPair;
  typedef std::pair<uint32_t, uint32_t> ClockDiff;
  typedef std::map<ExperimentPair, ClockDiff> ExperimentClockDiffMap;

  /// constructor, optional clock drift map can be passed in
  LessStreamDgram(const boost::shared_ptr<ExperimentClockDiffMap> expClockDiff = 
                  boost::shared_ptr<ExperimentClockDiffMap>(),
                  unsigned maxClockDrift = 120);

  /// operator < for two StreamDgram's
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
  bool lessClockCmp(const StreamDgram &a, const StreamDgram &b) const;
  bool lessFidCmp(const StreamDgram &a, const StreamDgram &b) const;
  bool lessBlockCmp(const StreamDgram &a, const StreamDgram &b) const;
  bool lessMapCmp(const StreamDgram &a, const StreamDgram &b) const;
 private:
  const boost::shared_ptr<ExperimentClockDiffMap> m_expClockDiff;
  std::map<DgramCategoryAB, CompareMethod> m_LUT;
  FiducialsCompare m_fidCompare;

 public:
 // UnknownCmp is thrown when two dgrams are compared for which no comparision
 // method is known - most likely an internal logic error in LessDgramCmp
 class UnknownCmp : public psana::Exception {
 public:
 UnknownCmp(const ErrSvc::Context &ctx) : 
   psana::Exception(ctx, "unknown cmp method - LessDgramCmp") {}
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
