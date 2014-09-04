#ifndef XTCINPUT_XTCSTREAMMERGER_H
#define XTCINPUT_XTCSTREAMMERGER_H

//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class XtcStreamMerger.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <queue>
#include <deque>
#include <boost/utility.hpp>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/StreamDgram.h"
#include "XtcInput/StreamFileIterI.h"
#include "XtcInput/XtcStreamDgIter.h"
#include "XtcInput/XtcFileName.h"
#include "XtcInput/XtcFilesPosition.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief datagram iterator which merges data from several streams.
 *
 *  Class responsible for merging of the datagrams from several
 *  XTC streams.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcStreamMerger : boost::noncopyable {
public:

  /**
   *  @brief Make iterator instance
   *
   *  @param[in]  streamIter  Iterator for input files
   *  @param[in]  l1OffsetSec Time offset to add to non-L1Accept transitions.
   *  @param[in]  firstControlStream starting stream number for fiducial merge
   *  @param[in]  maxStreamClockDiffSec maximum difference between stream clocks in seconds
   *              should be <= 85 seconds.
   *  @param[in]  thirdEvent if non-null, offsets for second event
   */
  XtcStreamMerger(const boost::shared_ptr<StreamFileIterI>& streamIter,
                  double l1OffsetSec, int firstControlStream,
                  unsigned maxStreamClockDiffSec,
                  boost::shared_ptr<XtcFilesPosition> thirdEvent) ;

  // Destructor
  ~XtcStreamMerger () ;

  /**
   *  @brief Return next datagram.
   *
   *  Read next datagram, return zero pointer after last file has been read,
   *  throws exception for errors.
   *
   *  @return Shared pointer to datagram object
   *
   *  @throw FileOpenException Thrown in case chunk file cannot be open.
   *  @throw XTCReadException Thrown for any read errors
   *  @throw XTCLiveTimeout Thrown for timeout during live data reading
   */
  Dgram next() ;

protected:

  // update time in datagram
  void updateDgramTime(Pds::Dgram& dgram) const ;

  // Struct to keep track of previous transition, L1 block number, and run. 
  struct TransBlock {
    Pds::TransitionId::Value trans;
    uint64_t block;
    int run;
    TransBlock() : trans(Pds::TransitionId::Unknown)
                   , block(0)
                   , run(0) 
    {};
    TransBlock(Pds::TransitionId::Value _trans, uint64_t _block, int _run) : 
         trans(_trans)
       , block(_block)
       , run(_run) 
    {};
    TransBlock(const TransBlock &o) : trans(o.trans)
                                      , block(o.block)
                                      , run(o.run) 
    {};
    TransBlock & operator=(const TransBlock &o) 
    { 
      trans = o.trans; 
      block = o.block; 
      run = o.run; 
      return *this; 
    }
  };

  // utilities for managing TransBlock
  static TransBlock makeTransBlock(const Dgram &dg, uint64_t block);
  static TransBlock getInitialTransBlock(const Dgram &dg);
  static uint64_t getNextBlock(const TransBlock & prevTransBlock, const Dgram &dg);
  bool processingDAQ() const { return m_processingDAQ; }

private:
  typedef std::pair<StreamDgram::StreamType, int> StreamIndex;
  static std::string dumpStr(const StreamIndex &streamIndex);           ///< debugging string for StreamIndex
  std::map<StreamIndex, boost::shared_ptr<XtcStreamDgIter> > m_streams; ///< Set of datagram iterators for streams
  std::map<StreamIndex, TransBlock> m_priorTransBlock;                  ///< TransBlock for last dgram from each stream

  bool m_processingDAQ;                       ///< set to true if DAQ streams exist in the merge
  int32_t m_l1OffsetSec ;                     ///< Time offset to add to non-L1Accept transitions (seconds)
  int32_t m_l1OffsetNsec ;                    ///< Time offset to add to non-L1Accept transitions (nanoseconds)
  int m_firstControlStream ;                  ///< starting stream number for control streams
  StreamDgramGreater m_streamDgramGreater;    ///< for comparing two dgrams in the priority queue
  boost::shared_ptr<XtcFilesPosition> m_thirdEvent; ///< if non-null, offsets for third event

  typedef std::priority_queue<StreamDgram, std::vector<StreamDgram>, StreamDgramGreater> OutputQueue;
  OutputQueue m_outputQueue;                  ///< Output queue for datagrams
};

} // namespace XtcInput

#endif // XTCINPUT_XTCSTREAMMERGER_H
