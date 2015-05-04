//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class XtcStreamMerger...
//
// Author List:
//     Andrei Salnikov
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
#include <sstream>
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

#define DBGMSG debug

using namespace XtcInput;

namespace {

const char* logger = "XtcInput.XtcStreamMerger" ;

bool isDisable(const XtcInput::Dgram &dg) {
  if (dg.empty()) return false;
  Pds::TransitionId::Value nextService = dg.dg()->seq.service();
  return (nextService == Pds::TransitionId::Disable);
}

boost::shared_ptr<XtcStreamDgIter::ThirdDatagram> 
checkForThirdDatagram(int stream, boost::shared_ptr<XtcFilesPosition> thirdEvent) {

  if (not thirdEvent) {
    MsgLog(logger, DBGMSG, "XtcStreamMerger: no third event position");
    return boost::shared_ptr<XtcStreamDgIter::ThirdDatagram>();
  }
  if (not thirdEvent->hasStream(stream)) {
    std::stringstream msg;
    msg << stream;
    throw StreamNotInPosition(ERR_LOC, msg.str());
  }
  std::pair<XtcFileName, off64_t> thirdDatagramThisStream = thirdEvent->getChunkFileOffset(stream);
  boost::shared_ptr<XtcStreamDgIter::ThirdDatagram> thirdDgram = 
    boost::make_shared<XtcStreamDgIter::ThirdDatagram>(thirdDatagramThisStream.first,
                                                        thirdDatagramThisStream.second);
  return thirdDgram;
}
  
} // local namespace

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
XtcStreamMerger::XtcStreamMerger(const boost::shared_ptr<StreamFileIterI>& streamIter,
                                 double l1OffsetSec, int firstControlStream,
                                 unsigned maxStreamClockDiffSec,
                                 boost::shared_ptr<XtcFilesPosition> thirdEvent) 
  : m_streams()
  , m_priorTransBlock()
  , m_processingDAQ(false)
  , m_l1OffsetSec(int(l1OffsetSec))
  , m_l1OffsetNsec(int((l1OffsetSec-m_l1OffsetSec)*1e9))
  , m_firstControlStream(firstControlStream)
  , m_streamDgramGreater(maxStreamClockDiffSec)
  , m_thirdEvent(thirdEvent)
  , m_outputQueue(m_streamDgramGreater)
    
{

  // create all streams
  int idxDAQ = 0;
  int idxCtrl = 0;
  while (true) {
    const boost::shared_ptr<ChunkFileIterI>& chunkFileIter = streamIter->next();
    if (not chunkFileIter) break;

    bool controlStream = int(streamIter->stream()) >= m_firstControlStream;
    bool clockSort = not controlStream;

    boost::shared_ptr<XtcStreamDgIter::ThirdDatagram> thirdDatagram = 
      checkForThirdDatagram(streamIter->stream(), m_thirdEvent);

    // create new stream
    const boost::shared_ptr<XtcStreamDgIter>& stream = 
      boost::make_shared<XtcStreamDgIter>(chunkFileIter, thirdDatagram, clockSort);
    if (controlStream) {
      StreamDgram dg(stream->next(), StreamDgram::controlUnderDAQ, 0, idxCtrl);
      StreamIndex streamIndex(StreamDgram::controlUnderDAQ, idxCtrl);
      ++idxCtrl;
      m_streams[streamIndex] = stream;
      m_priorTransBlock[streamIndex] = getInitialTransBlock(dg);
      if (not dg.empty()) updateDgramTime(*dg.dg());
      m_outputQueue.push(dg);
      MsgLog(logger, DBGMSG, "XtcStreamMerger initialization. Added " 
             << StreamDgram::dumpStr(dg)); 
    } else {
      // this is a DAQ stream
      StreamDgram dg(stream->next(), StreamDgram::DAQ, 0, idxDAQ);
      StreamIndex streamIndex(StreamDgram::DAQ, idxDAQ);
      ++idxDAQ;
      m_streams[streamIndex] = stream;
      m_priorTransBlock[streamIndex] = getInitialTransBlock(dg);
      if (not dg.empty()) updateDgramTime(*dg.dg());
      m_outputQueue.push(dg);
      MsgLog(logger, DBGMSG, "XtcStreamMerger initialization. Added " 
             << StreamDgram::dumpStr(dg));
    }
  }
  if (idxDAQ > 0) {
    m_processingDAQ = true;
  }
  MsgLog(logger, DBGMSG, "XtcStreamMerger initialization: "
         << idxDAQ << " DAQ streams and " << idxCtrl << " control streams");
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
  if (m_outputQueue.empty()) return Dgram();

  StreamDgram nextStreamDg = m_outputQueue.top();
  m_outputQueue.pop();
  int replaceStreamId = nextStreamDg.streamId();
  StreamIndex replaceStreamIndex(nextStreamDg.streamType(), replaceStreamId);
    
  if (m_streams.find(replaceStreamIndex) == m_streams.end()) {
    throw psana::Exception(ERR_LOC, "XtcStreamMerger::next() replacement stream index not found in m_streams");
  }
  if (m_priorTransBlock.find(replaceStreamIndex) == m_priorTransBlock.end()) {
    throw psana::Exception(ERR_LOC, "XtcStreamMerger::next() replacement stream index not found in m_priorTransBlock");
  }

  MsgLog(logger,DBGMSG,"next() returning: " << StreamDgram::dumpStr(nextStreamDg));

  bool replaced = false;
  while (not replaced) {
    Dgram replaceDg = m_streams[replaceStreamIndex]->next();
    TransBlock lastTransBlock = m_priorTransBlock[replaceStreamIndex];
    uint64_t replaceBlock = getNextBlock(lastTransBlock, replaceDg);
    m_priorTransBlock[replaceStreamIndex] = makeTransBlock(replaceDg, replaceBlock);

    // skip over enable and disable transitions in all streams. We use EndCalibCycle for 
    // the L1Block count as its synchronization across the DAQ streams is more robust.
    // Sending the Enable/Disable pairs that exist in between Begin/End Calib cycle's 
    // through the prioriry queue is Ok, but it is cleaner to keep them out.
    // Downstream processing in PSXtcInput does not use Disable/Enable.

    // For control streams, if we are processing DAQ streams, skip over all transitions.
    // they should contain no event data. However if we are only processing control streams,
    // process the control transitions.

    // Skip over L1Accepts that have 0x1FFFF as the fiducial value. In older data, the control streams 
    // were producing these when they had trouble getting the DAQ timestamp. These control 
    // stream Dgrams cannot be merged and will trigger warnings from FiducialsCompare.

    bool skip = false;
    if (not replaceDg.empty()) {
      if ( (replaceDg.dg()->seq.service() == Pds::TransitionId::Enable) or
           (replaceDg.dg()->seq.service() == Pds::TransitionId::Disable) ) {
        MsgLog(logger, DBGMSG, "next() skipping Enable or Disable in " 
               << dumpStr(replaceStreamIndex));
        skip = true;
      } else if ( (replaceDg.dg()->seq.service() == Pds::TransitionId::L1Accept) and
                  (replaceDg.dg()->seq.stamp().fiducials() >= Pds::TimeStamp::MaxFiducials) ) {
        MsgLog(logger, DBGMSG, "next() skipping L1Accept with fiducials >= " 
               << Pds::TimeStamp::MaxFiducials << " in " 
               << dumpStr(replaceStreamIndex));
        skip = true;
      }
    } else if (processingDAQ()) {
      if ((nextStreamDg.streamType() == StreamDgram::controlUnderDAQ) or 
          (nextStreamDg.streamType() == StreamDgram::controlIndependent)) {
        if (not replaceDg.empty()) {
          Pds::TransitionId::Value replaceTrans = replaceDg.dg()->seq.service();
          if (replaceTrans == Pds::TransitionId::Configure) {
            // the first configure was put in the queue during initialization. Now we have a
            // configure in the midst of the stream.
            MsgLog(logger, warning, "Discarding Configure transition found in " 
                   << replaceDg.file().path()
                   << " expected if processing multiple runs, investigate further if not");
          }
          if (replaceTrans != Pds::TransitionId::L1Accept) {
            MsgLog(logger, DBGMSG, "next() skipping non L1Accept in " 
                   << dumpStr(replaceStreamIndex));
            skip = true;
          }
        }
      }
    }
    if (not skip) {
      StreamDgram replaceStreamDg(replaceDg, nextStreamDg.streamType(), replaceBlock, replaceStreamId);
      m_outputQueue.push(replaceStreamDg);
      replaced = true;
    }
  }
  return nextStreamDg;
}

// updates the time for non L1 Accepts
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
    return TransBlock();
  }
  return TransBlock(dg.dg()->seq.service(), block, dg.file().run());
}

XtcStreamMerger::TransBlock XtcStreamMerger::getInitialTransBlock(const Dgram &dg) {
  return makeTransBlock(dg, 0);
}

uint64_t  XtcStreamMerger::getNextBlock(const TransBlock & prevTransBlock, const Dgram &dg) {
  if (dg.empty()) {
    return prevTransBlock.block;
  }
  int nextRun = dg.file().run();
  if (nextRun != prevTransBlock.run) return 0;
  Pds::TransitionId::Value nextService = dg.dg()->seq.service();
  // increment the block count if we see a EndCalibCycle, and the prior transition was not also a
  // EndCalibCycle. generally there should not be two EndCalibCycle's in a row, this protects against 
  // problems in the data
  if ((prevTransBlock.trans != Pds::TransitionId::EndCalibCycle) and 
      (nextService == Pds::TransitionId::EndCalibCycle)) {
    return prevTransBlock.block + 1;
  }
  return prevTransBlock.block;
}

std::string XtcStreamMerger::dumpStr(const StreamIndex &streamIndex) {
  std::ostringstream msg;
  msg << "streamType=" 
      << StreamDgram::streamType2str(streamIndex.first) 
      << " streamId=" << std::setw(2) << streamIndex.second;
  return msg.str();
} 

} // namespace XtcInput
