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
      
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcStreamMerger::XtcStreamMerger(const boost::shared_ptr<StreamFileIterI>& streamIter,
                                 double l1OffsetSec, int firstControlStream)
  : m_DAQstreams()
  , m_controlStreams()
  , m_DAQ_priorTransBlock()
  , m_control_priorTransBlock()
  , m_l1OffsetSec(int(l1OffsetSec))
  , m_l1OffsetNsec(int((l1OffsetSec-m_l1OffsetSec)*1e9))
  , m_firstControlStream(firstControlStream)
  , m_streamDgramCmp()
  , m_outputQueue(m_streamDgramCmp)
{

  // create all streams
  int daqStreamIndex = 0;
  int controlStreamIndex = 0;
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
    if (controlStream) {
      StreamDgram dg(stream->next(), controlUnderDAQ, 0, controlStreamIndex++);
      m_controlStreams.push_back(stream);
      m_control_priorTransBlock.push_back(getInitialTransBlock(dg));
      m_outputQueue.push(dg);
    } else {
      StreamDgram dg(stream->next(), DAQ, 0, daqStreamIndex++);
      m_DAQstreams.push_back(stream) ;
      m_DAQ_priorTransBlock.push_back(getInitialTransBlock(dg));
      // only adjust times of dgrams for typical streams, this allows one to
      // correct for too much clock drift with fiducial merge streams
      if (not dg.empty()) updateDgramTime(*dg.dg());
      m_outputQueue.push(dg);
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
  if (not m_outputQueue.empty()) {
    StreamDgram nextStreamDg = m_outputQueue.top();
    m_outputQueue.pop();
    int replaceStreamIdx = nextStreamDg.streamIndex();
    Dgram replaceDg;
    TransBlock lastTransBlock;
    uint64_t replaceBlock=0;
    switch (nextStreamDg.streamType()) {
    case DAQ:
      replaceDg = m_DAQstreams[replaceStreamIdx]->next();
      lastTransBlock = m_DAQ_priorTransBlock[replaceStreamIdx];
      replaceBlock = getNextBlock(lastTransBlock, replaceDg);
      m_DAQ_priorTransBlock[replaceStreamIdx] = makeTransBlock(replaceDg, replaceBlock);
      break;
    case controlUnderDAQ:
      replaceDg = m_controlStreams[replaceStreamIdx]->next();
      lastTransBlock = m_control_priorTransBlock[replaceStreamIdx];
      replaceBlock = getNextBlock(lastTransBlock, replaceDg);
      m_control_priorTransBlock[replaceStreamIdx] = makeTransBlock(replaceDg, replaceBlock);
      break;
    case controlIndependent:
      throw psana::Exception(ERR_LOC, "XtcStreamMerger::next() controlIndependent not implemented");
      break;
    }
    StreamDgram replaceStreamDg(replaceDg, nextStreamDg.streamType(), replaceBlock, replaceStreamIdx);
    m_outputQueue.push(replaceStreamDg);
    return nextStreamDg;
  }
  return Dgram();
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

XtcStreamMerger::TransBlock XtcStreamMerger::makeTransBlock(const Dgram &dg, uint64_t block) {
  if (dg.empty()) {
    return TransBlock(Pds::TransitionId::Unknown, block);
  }
  return TransBlock(dg.dg()->seq.service(), block);
}

XtcStreamMerger::TransBlock XtcStreamMerger::getInitialTransBlock(const Dgram &dg) {
  return makeTransBlock(dg, 0);
}

uint64_t  XtcStreamMerger::getNextBlock(const TransBlock & prevTransBlock, const Dgram &dg) {
  if (dg.empty()) {
    return prevTransBlock.second;
  }
  Pds::TransitionId::Value nextService = dg.dg()->seq.service();
  if ((prevTransBlock.first != Pds::TransitionId::L1Accept) and (nextService == Pds::TransitionId::L1Accept)) {
    return prevTransBlock.second + 1;
  }
  return prevTransBlock.second;
}
  
} // namespace XtcInput
