//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcStreamMerger...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/XtcStreamMerger.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <map>
#include <iomanip>
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/ChunkFileIterList.h"
#include "XtcInput/Exceptions.h"
#include "pdsdata/xtc/TransitionId.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "XtcInput.XtcStreamMerger" ;

  const int MAX_CLOCK_DRIFT_SECONDS = 120;

  // return index of dgramList with earliest clocktime, or -1 if all
  // dgrams are empty
  int findEarliestClockTime(const std::vector<XtcInput::Dgram> &dgrams) {
    unsigned n = dgrams.size();
    int stream = -1;
    for ( unsigned i = 0 ; i < n ; ++ i ) {
      if (not dgrams[i].empty()) {
        if ( stream < 0 or 
             dgrams[stream].dg()->seq.clock() > dgrams[i].dg()->seq.clock() ) {
          stream = i ;
        }
      }
    }
    return stream;
  }

  // return index of dgram that is an L1Accept with earliest fiducial time, or -1 if no 
  // L1Accepts in dgram list
  int findEarliestL1AcceptFidTime(const std::vector<XtcInput::Dgram> &dgrams,
                                  const XtcInput::FiducialsCompare &fidCmp) {
    unsigned n = dgrams.size();
    int stream = -1;
    for ( unsigned i = 0 ; i < n ; ++ i ) {
      if (not dgrams[i].empty() and (dgrams[i].dg()->seq.service() == Pds::TransitionId::L1Accept)) {
        if (stream < 0 or fidCmp.fiducialsGreater(*dgrams[stream].dg(), *dgrams[i].dg())) {
          stream = i ;
        }
      }
    }
    return stream;
  }

  float clockTimeDiffInSeconds(XtcInput::Dgram::ptr A, XtcInput::Dgram::ptr B) {
    float res = A->seq.clock().seconds() - B->seq.clock().seconds();
    float nanoDiff = A->seq.clock().nanoseconds() - B->seq.clock().nanoseconds();
    nanoDiff /= 1e9;
    res += nanoDiff;
    return res;
  }

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

const unsigned XtcStreamMerger::maxSizeOutputQueue = 100;

//----------------
// Constructors --
//----------------
XtcStreamMerger::XtcStreamMerger(const boost::shared_ptr<StreamFileIterI>& streamIter,
                                 double l1OffsetSec, int firstControlStream)
  : m_DAQstreams()
  , m_DAQdgrams()
  , m_controlStreams()
  , m_controlDgrams()
  , m_l1OffsetSec(int(l1OffsetSec))
  , m_l1OffsetNsec(int((l1OffsetSec-m_l1OffsetSec)*1e9))
  , m_firstControlStream(firstControlStream)
  , m_outputQueue()
  , m_fidCmp(120)
{

  // create all streams
  while (true) {
    const boost::shared_ptr<ChunkFileIterI>& chunkFileIter = streamIter->next();
    if (not chunkFileIter) break;

    bool controlStream = int(streamIter->stream()) >= m_firstControlStream;
    bool clockSort = not controlStream;
    MsgLog(logger, trace, "XtcStreamMerger -- stream: " << streamIter->stream()
           << " is control stream=" << controlStream);

    // create new stream
    const boost::shared_ptr<XtcStreamDgIter>& stream = 
      boost::make_shared<XtcStreamDgIter>(chunkFileIter, clockSort) ;
    Dgram dg(stream->next());
    if (controlStream) {
      m_controlStreams.push_back(stream) ;
      m_controlDgrams.push_back(dg);
    } else {
      m_DAQstreams.push_back(stream) ;
      // only adjust times of dgrams for typical streams, this allows one to
      // correct for too much clock drift with fiducial merge streams
      if (not dg.empty()) updateDgramTime(*dg.dg());
      m_DAQdgrams.push_back( dg ) ;
    }
  }

}

//--------------
// Destructor --
//--------------
XtcStreamMerger::~XtcStreamMerger ()
{
}

// read next datagram, return zero pointer after last file has been read,
// throws exception for errors.
Dgram
XtcStreamMerger::next()
{

  if (m_outputQueue.empty()) {

    // find datagram with lowest timestamp
    int DAQstream = findEarliestClockTime(m_DAQdgrams);

    MsgLog( logger, debug, "next -- DAQ stream: " << DAQstream ) ;

    if (DAQstream >= 0) {
      XtcInput::Dgram::ptr DAQdg = m_DAQdgrams[DAQstream].dg();
      Pds::TransitionId::Value targetTrans = DAQdg->seq.service();
      if (targetTrans == Pds::TransitionId::Configure) {
        int controlStream = findEarliestL1AcceptFidTime(m_controlDgrams, fidCmp());
        if (controlStream >= 0) {
          XtcInput::Dgram::ptr controlDg = m_controlDgrams[controlStream].dg();
          if (clockTimeDiffInSeconds(controlDg, DAQdg) > MAX_CLOCK_DRIFT_SECONDS) {
            MsgLog(logger, error, "configure transition in DAQ streams, but next control streams include L1Accept more than " << MAX_CLOCK_DRIFT_SECONDS << " in the future");
          }
          sendFidMatchL1AcceptsToOutputQueue(controlDg, true);
        }
      }
          
      // send all datagrams with this timestamp to output queue
      Pds::ClockTime ts = m_DAQdgrams[DAQstream].dg()->seq.clock();
      bool isControl = false;
      sendClockMatchToOutputQueue(ts, isControl);
      
    }

  }

  Dgram dg;
  if (not m_outputQueue.empty()) {
    // this datagram will be returned
    dg = m_outputQueue.front();
    m_outputQueue.pop();

    Dgram::ptr dgptr = dg.dg();
    MsgLog( logger, debug, "next -- m_DAQdgrams[stream].clock: "
        << dgptr->seq.clock().seconds() << " sec " << dgptr->seq.clock().nanoseconds() << " nsec" ) ;
    MsgLog( logger, debug, "next -- m_DAQdgrams[stream].service: " << Pds::TransitionId::name(dgptr->seq.service()) ) ;
  }

  return dg ;
}

int XtcStreamMerger::sendClockMatchToOutputQueue(const Pds::ClockTime  &ts, 
                                                 bool control) {
  std::vector<Dgram> &dgrams = control ? m_controlDgrams : m_DAQdgrams;
  std::vector<boost::shared_ptr<XtcStreamDgIter> > &streams = control ? m_controlStreams : m_DAQstreams;
  int numSent = 0;
  unsigned n = dgrams.size();
  for ( unsigned i = 0 ; i < n ; ++ i ) {
    if (not dgrams[i].empty()) {
      if (dgrams[i].dg()->seq.clock() == ts) {
        m_outputQueue.push(dgrams[i]);
        ++numSent;
        // get next datagram from that stream
        Dgram ndg(streams[i]->next());
        MsgLog( logger, debug, " read datagram from file: " << ndg.file().basename() ) ;
        if (not ndg.empty() and (not control)) updateDgramTime(*ndg.dg());
        dgrams[i] = ndg ;
      }
    }
  }
  return numSent;
}

int XtcStreamMerger::sendFidMatchL1AcceptsToOutputQueue(XtcInput::Dgram::ptr dg, 
                                                        bool control) {
  std::vector<Dgram> &dgrams = control ? m_controlDgrams : m_DAQdgrams;
  std::vector<boost::shared_ptr<XtcStreamDgIter> > &streams = control ? m_controlStreams : m_DAQstreams;
  int numSent = 0;
  unsigned n = dgrams.size();
  for ( unsigned i = 0 ; i < n ; ++ i ) {
    if (not dgrams[i].empty()) {
      if (fidCmp().fiducialsEqual(*dgrams[i].dg(), *dg)) {
        m_outputQueue.push(dgrams[i]);
        ++numSent;
        // get next datagram from that stream
        Dgram ndg(streams[i]->next());
        MsgLog( logger, debug, " read datagram from file: " << ndg.file().basename() ) ;
        if (not ndg.empty() and (not control)) updateDgramTime(*ndg.dg());
        dgrams[i] = ndg ;
      }
    }
  }
  return numSent;
}

void 
XtcStreamMerger::updateDgramTime(Pds::Dgram& dgram) const
{
  if ( dgram.seq.service() != Pds::TransitionId::L1Accept ) {

    // update clock values
    const Pds::ClockTime& time = dgram.seq.clock() ;
    int32_t sec = time.seconds() + m_l1OffsetSec;
    int32_t nsec = time.nanoseconds() + m_l1OffsetNsec;
    if (nsec < 0) {
        nsec += 1000000000;
        -- sec;
    } else if (nsec >= 1000000000) {
        nsec -= 1000000000;
        ++ sec;
    }      
    Pds::ClockTime newTime(sec, nsec) ;

    // there is no way to change clock field in datagram but there is 
    // an assignment operator
    dgram.seq = Pds::Sequence(newTime, dgram.seq.stamp());
  }
}

} // namespace XtcInput
