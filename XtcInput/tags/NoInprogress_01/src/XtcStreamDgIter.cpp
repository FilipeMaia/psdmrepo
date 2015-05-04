//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class XtcStreamDgIter...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcStreamDgIter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/Exceptions.h"
#include "XtcInput/XtcChunkDgIter.h"
#include "pdsdata/xtc/Xtc.hh"
#include "XtcInput/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "XtcInput.XtcStreamDgIter" ;

  // size of the read-ahead buffer
  const unsigned readAheadSize = 64;

  // functor to match header against specified clock time
  struct MatchClock {
    MatchClock(const Pds::ClockTime& clock) : m_clock(clock) {}
    bool operator()(const boost::shared_ptr<XtcInput::DgHeader>& header) const {
      return header->clock() == m_clock;
    }

    Pds::ClockTime m_clock;
  };

  // function to compare xtc files
  bool xtcFilesNotEqual(const XtcInput::XtcFileName &a, const XtcInput::XtcFileName &b) {
    int chunkA = a.chunk();
    int chunkB = b.chunk();
    bool equalChunks = chunkA == chunkB;
    bool notEqualPaths  = false;
    if (a < b) {
      notEqualPaths = true;
    }
    if (b < a) {
      notEqualPaths = true;
    }
    if (equalChunks and notEqualPaths) {
      MsgLog(logger,warning, "xtc files are not equal but chunks are: a=" << a << " b=" << b);
    }
    return notEqualPaths;
  }
}

//             ----------------------------------------
//             -- Public Function Member Definitions --
//             ----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcStreamDgIter::XtcStreamDgIter(const boost::shared_ptr<ChunkFileIterI>& chunkIter,
                                 bool clockSort)
  : m_chunkIter(chunkIter)
  , m_dgiter()
  , m_chunkCount(0)
  , m_streamCount(0)
  , m_headerQueue()
  , m_clockSort(clockSort)
{
  m_headerQueue.reserve(::readAheadSize);
}

XtcStreamDgIter::XtcStreamDgIter(const boost::shared_ptr<ChunkFileIterI>& chunkIter,
                                 const boost::shared_ptr<ThirdDatagram> & thirdDatagram,
                                 bool clockSort)
  : m_chunkIter(chunkIter)
  , m_dgiter()
  , m_chunkCount(0)
  , m_streamCount(0)
  , m_headerQueue()
  , m_clockSort(clockSort)
  , m_thirdDatagram(thirdDatagram)
{
  m_headerQueue.reserve(::readAheadSize);
}
//--------------
// Destructor --
//--------------
XtcStreamDgIter::~XtcStreamDgIter ()
{
}

// read next datagram, return zero pointer after last file has been read,
// throws exception for errors.
Dgram
XtcStreamDgIter::next()
{
  // call other method to fill up and sort the queue
  readAhead();

  // pop one datagram if queue is not empty
  Dgram dgram;
  while (not m_headerQueue.empty()) {
    boost::shared_ptr<DgHeader> hptr = m_headerQueue.front();
    m_headerQueue.erase(m_headerQueue.begin());
    Dgram::ptr dg = hptr->dgram();
    if (dg) {
      dgram = Dgram(dg, hptr->path(), hptr->offset());
      break;
    } else {
      // header failed to read datagram, this is likely due to non-fatal
      // error like premature EOF. Skip this one and try to go to the next
      readAhead();
    }
  }

  return dgram ;
}

// fill the read-ahead queue
void
XtcStreamDgIter::readAhead()
{

  while (m_headerQueue.size() < ::readAheadSize) {

    if (not m_dgiter) {

      // get next file name
      const XtcFileName& file = m_chunkIter->next();

      // if no more file then stop
      if (file.path().empty()) break ;

      // open next xtc file if there is none open
      MsgLog(logger, trace, "processing file: " << file) ;
      m_dgiter = boost::make_shared<XtcChunkDgIter>(file, m_chunkIter->liveTimeout());
      m_chunkCount = 0 ;
    }

    boost::shared_ptr<DgHeader> hptr; // next datagram header

    // check for special parameters to jump for the third datagram 
    if (m_streamCount == 2) {    // we are at the third datagram, streamCount is 0 based
      if (m_thirdDatagram) {
        const XtcFileName & xtcFileForThirdDgram = m_thirdDatagram->xtcFile;
        off64_t offsetForThirdDgram = m_thirdDatagram->offset;
        MsgLog(logger,debug,"third datagram jump: will try to jump to offset=" 
	       << offsetForThirdDgram << " in file=" << xtcFileForThirdDgram); 
        // advance chunk file if need be
        while (xtcFilesNotEqual(m_dgiter->path(), xtcFileForThirdDgram)) {
          MsgLog(logger,debug,"third datagram jump, jmpFile != currentFile - "
		 << xtcFileForThirdDgram << " != " << m_dgiter->path());
          // get next file name
          const XtcFileName& file = m_chunkIter->next();
          
          if (file.path().empty()) {
	    // we went through all the files in the chunkIter
            throw FileNotInStream(ERR_LOC, xtcFileForThirdDgram.path());
          }
          // open file
          MsgLog(logger, trace, " looking for third dgram - opening file: " << file) ;
          m_dgiter = boost::make_shared<XtcChunkDgIter>(file, m_chunkIter->liveTimeout());
          m_chunkCount = 0;
        }
        hptr = m_dgiter->nextAtOffset(offsetForThirdDgram);
      } else {
        // this is the third datagram (streamCount == 2), but no special jump to do
        MsgLog(logger,debug,"third datagram, no jmp");
        hptr = m_dgiter->next();
      }
    } else {
      // typical case, streamCount != 1
      hptr = m_dgiter->next();
    }

    // if failed to read go to next file
    if (not hptr) {
      m_dgiter.reset();
    } else {
      // read full datagram
      queueHeader(hptr);
      ++ m_chunkCount ;
      ++ m_streamCount ;
    }

  }

  MsgLog(logger, debug, "headers queue has size " << m_headerQueue.size()) ;

}

// add one header to the queue in a correct position based on clockSort
void
XtcStreamDgIter::queueHeader(const boost::shared_ptr<DgHeader>& header)
{
  Pds::TransitionId::Value tran = header->transition();
  const Pds::ClockTime& clock = header->clock();
  MsgLog(logger, debug, "XtcStreamDgIter::queueHeader: transition: " << Pds::TransitionId::name(tran)
         << " time: " << clock.seconds() << "sec " << clock.nanoseconds() << " nsec"
         << " clockSort=" << m_clockSort);

  if (m_clockSort) {  
    // For split transitions look at the queue and find matching split transition,
    // store them together if found, otherwise assume it's first piece and store
    // it like normal transition.
    if (header->damage().value() & (1 << Pds::Damage::DroppedContribution)) {
      HeaderQueue::iterator it = std::find_if(m_headerQueue.begin(), m_headerQueue.end(), MatchClock(clock));
      if (it != m_headerQueue.end()) {
        MsgLog(logger, debug, "XtcStreamDgIter::queueHeader: split transition, found match");
        m_headerQueue.insert(it, header);
        return;
      }
    }
  }

  // At this point we have either clockSort=false or clockSort=True and a 
  // non-split transition or split transition without
  // other pieces of the same split transitions (other pieces may have appeared
  // already but have been popped from queue then). We treat single piece of split
  // transition the same as non-split transition below.

  // There was time when clocks for L1Accept transitions and other transition types
  // were separate and sometimes not synchronized. Without applying clock drift
  // correction we still want to keeps things stable here, so we do not move
  // L1Accept and non-L1Accept transitions w.r.t. each other and we do not re-arrange
  // non-split transitions w.r.t each other. We can re-arrange L1Accept transitions
  // but only if they are in the same run of L1Accepts (there are no non-L1Accept
  // transitions between two L1Accepts).

  if (tran != Pds::TransitionId::L1Accept) {
    MsgLog(logger, debug, "XtcStreamDgIter::queueHeader: non-event transition, append");
    m_headerQueue.push_back(header);
    return;
  }

  /*
   *  At this point we have L1Accept. If clockSort, start from the end of the 
   *  queue and walk to the head until we meet either earlier L1Accept or 
   * non-L1Accept transition. If not clockSort, place at the end of the queue.
   */
  if (not m_clockSort) {
    m_headerQueue.push_back(header);
  } else {
    for (HeaderQueue::iterator it = m_headerQueue.end(); it != m_headerQueue.begin(); -- it) {
      const boost::shared_ptr<DgHeader>& prev = *(it - 1);

      if (prev->transition() != Pds::TransitionId::L1Accept) {
        MsgLog(logger, debug, "XtcStreamDgIter::queueHeader: insert L1Accept after non-L1Accept");
        m_headerQueue.insert(it, header);
        return;
      } else if (clock > prev->clock()) {
        MsgLog(logger, debug, "XtcStreamDgIter::queueHeader: insert L1Accept after earlier L1Accept");
        m_headerQueue.insert(it, header);
        return;
      }
    }

    // could not find any acceptable place, means this transition is earlier than all
    // other transitions, add it to the head of the queue
    MsgLog(logger, debug, "XtcStreamDgIter::queueHeader: insert L1Accept at the queue head");
    m_headerQueue.insert(m_headerQueue.begin(), header);
  }
    
}

} // namespace XtcInput
