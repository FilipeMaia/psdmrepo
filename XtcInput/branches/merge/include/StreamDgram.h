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

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Dgram with attached data for sorting.
 *
 *  Adds a stream type, L1block number, and streamId to a Dgram.
 *  The stream type and L1block number are used for sorting. It is up to clients
 *  of StreamDgram to initialize the L1block with the correct value.
 *
 *  L1block number -
 *  If this dgram is an L1Accept, L1Block returns a 0-up counter that is the
 *  block of L1Accepts that this Dgram is a part of.  If this dgram is a 
 *  Transition, returns the number of L1Accept blocks that precede it.
 *  For example:
 *  dgram stream: T T L L L L L L T T L L L L T T L L T T L L L T L L T  
 *  L1Block:      0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4
 *
 *  The counter must increment for 'empty' L1Blocks, that can occur when a enable
 *  Transition is followed immediately by a disable transtion. 
 *
 *  The simplest way to compute the L1Block is to count the number of Disable
 *  transitions that have occurred in the stream - every block of L1Accepts (empty
 *  or not) ends with a Disable.
 *
 *  The L1Block is relative to a run. When the run changes, the L1Block should be
 *  reset to 0.
 *
 *  Within a run, the L1Block counts the number of disable transitions up to and 
 *  including the current transition.
 *
 *  From the above example, one sees that within a run, 
 *
 *           T > L     if      L1Block(T) > L1Block(L) 
 *           L > T     if      L1Block(L) >= L1Block(T) 
 *
 *  It is possible to change the definition of L1Block, for instance it could be a
 *  EndCalibCycle count rather than Disable count. In this case one should remove all
 *  Enable/Disable transitions from any comparison done with StreamDgramGreater below.
 *
 *  The streamId can be used by clients of the class for keeping track of the dgrams. 
 *  For instance, it could be the stream number so as to identify Dgrams that move 
 *  in and out of a priority queue.
 *
 *  @version $Id$
 *
 *  @author David Schneider
 */
class StreamDgram : public Dgram {
 public:
  typedef enum {DAQ, controlUnderDAQ, controlIndependent} StreamType;  

 StreamDgram(const Dgram &dgram, StreamType streamType, int64_t L1block, int streamId) 
   : Dgram(dgram), m_streamType(streamType), m_L1block(L1block), m_streamId(streamId)
  {}

  /**
   *  Default ctor
   */
 StreamDgram() : Dgram(), m_streamType(DAQ), m_L1block(-1) {}

  /// returns L1Block, or -1 if Dgram is empty.
  int64_t L1Block() const { 
    if (empty()) return -1; 
    return m_L1block; 
  };

  /// returns streamType
  StreamType streamType() const { return m_streamType; }

  /// returns streamId
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
class StreamDgramGreater {
 public:
  /// constructor, optional clock drift map can be passed in
  StreamDgramGreater(unsigned maxClockDrift = 85);

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
  typedef enum {clockGreater, fidGreater, blockGreater, badGreater} CompareMethod;  

  /**
   * @brief returns true if a > b based on the clock
   *
   * First the run and L1Block are used. For dgrams in the same run and L1Block,
   * the clock is used.
   */
  bool doClockGreater(const StreamDgram &a, const StreamDgram &b) const;

  /**
   * @brief returns true if a > b based on the sec/fiducials.
   *
   * First the run and L1Block are used. For dgrams in the same run and L1Block,
   * see the class FiducialsCompare.
   */
  bool doFidGreater(const StreamDgram &a, const StreamDgram &b) const;

  /**
   * @brief returns true if a > b based on the block number and dgram transition.
   *
   * The two datagrams must have mixed transitions, that is one must be an L1Accept,
   * and the other not an L1Accept. Otherwise an exception is thrown.
   */
  bool doBlockGreater(const StreamDgram &a, const StreamDgram &b) const;

  /**
   * @brief place holder for a not implemented compare.
   *
   * Throws an exception if called.
   */
  bool doBadGreater(const StreamDgram &a, const StreamDgram &b) const;

  /**
   *  @brief returns -1, 0, 1 based on comparing run numbers.
   *
   *  a < b: returns -1
   *  a==b : returns  0   
   *  a > b: returns  1
   * 
   *  where the run number is obtained from the dgram filename
   */
  int runLessGreater(const StreamDgram &a, const StreamDgram &b) const;

  /**
   *  @brief returns -1, 0, 1 based on comparing L1Block
   *
   *  a < b: returns -1
   *  a ==b : returns  0   
   *  a > b: returns  1
   */
  int blockLessGreater(const StreamDgram &a, const StreamDgram &b) const;
 private:
  std::map<DgramCategoryAB, CompareMethod> m_LUT;
  FiducialsCompare m_fidCompare;

 public:
 // UnknownGreater is thrown when two dgrams are compared for which no comparision
 // method is known - most likely an internal logic error in StreamDgramGreater
 class UnknownGreater : public psana::Exception {
 public:
 UnknownGreater(const ErrSvc::Context &ctx) : 
   psana::Exception(ctx, "unknown cmp method - StreamDgramGreater") {}
 };
};

} // namespace XtcInput

#endif // XTCINPUT_STREAMDGRAM_H
