//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcStreamDgIter...
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
#include "XtcInput/FiducialsCompare.h"
#include "pdsdata/xtc/Xtc.hh"

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

  // functor to match header against specified Fiducials and clock time
  struct MatchFiducials {
    MatchFiducials(const Pds::ClockTime& clock, 
                   const unsigned fiducials,
                   const XtcInput::FiducialsCompare &fiducialsCompare) :
      m_clock(clock), m_fiducials(fiducials), 
      m_fiducialsCompare(fiducialsCompare) {}
    bool operator()(const boost::shared_ptr<XtcInput::DgHeader>& header) const {
      return m_fiducialsCompare.fiducialsEqual(header->clock().seconds(),
                                               header->fiducials(),
                                               m_clock.seconds(),
                                               m_fiducials);
    }

    Pds::ClockTime m_clock;
    unsigned m_fiducials;
    const XtcInput::FiducialsCompare & m_fiducialsCompare;
  };
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcStreamDgIter::XtcStreamDgIter(const boost::shared_ptr<ChunkFileIterI>& chunkIter, 
                                 const FiducialsCompare &fiducialsCompare)
  : m_chunkIter(chunkIter)
  , m_dgiter()
  , m_count(0)
  , m_headerQueue()
  , m_fiducialsCompare(fiducialsCompare)
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
      dgram = Dgram(dg, hptr->path());
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
      m_count = 0 ;
    }

    // try to read next event from it
    boost::shared_ptr<DgHeader> hptr = m_dgiter->next() ;

    // if failed to read go to next file
    if (not hptr) {
      m_dgiter.reset();
    } else {
      // read full datagram
      queueHeader(hptr);
      ++ m_count ;
    }

  }

  MsgLog(logger, debug, "headers queue has size " << m_headerQueue.size()) ;

}

// add one header to the queue in a correct position
void
XtcStreamDgIter::queueHeader(const boost::shared_ptr<DgHeader>& header)
{
  Pds::TransitionId::Value tran = header->transition();
  const Pds::ClockTime& clock = header->clock();
  const unsigned fiducials = header->fiducials();
  MsgLog(logger, debug, "XtcStreamDgIter::queueHeader: transition: " << Pds::TransitionId::name(tran)
         << " time: " << clock.seconds() << "sec " << clock.nanoseconds() << " nsec, "
         << " fiducials: " << fiducials);

  // For split transitions look at the queue and find matching split transition,
  // store them together if found, otherwise assume it's first piece and store
  // it like normal transition.
  if (header->damage().value() & (1 << Pds::Damage::DroppedContribution)) {
    HeaderQueue::iterator it = std::find_if(m_headerQueue.begin(), m_headerQueue.end(), 
                                            MatchFiducials(clock, fiducials, m_fiducialsCompare));
    if (it != m_headerQueue.end()) {
      MsgLog(logger, debug, "XtcStreamDgIter::queueHeader: split transition, found match");
      m_headerQueue.insert(it, header);
      return;
    }
  }

  // At this point we have either non-split transition or split transition without
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
   *  At this point we have L1Accept. Start from the end of the queue and walk to the
   *  head until we meet either earlier L1Accept or non-L1Accept transition.
   */
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

  // could not find any acceptable palce, means this transition is earlier than all
  // other transitions, add it to the head of the queue
  MsgLog(logger, debug, "XtcStreamDgIter::queueHeader: insert L1Accept at the queue head");
  m_headerQueue.insert(m_headerQueue.begin(), header);

}

} // namespace XtcInput
