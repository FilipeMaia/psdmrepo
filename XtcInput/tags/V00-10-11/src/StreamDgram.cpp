#include <iomanip>

#include "MsgLogger/MsgLogger.h"
#include "XtcInput/StreamDgram.h"

#define DBGMSG debug

namespace {
  const char *logger = "StreamDgram";

  std::string bool2str(bool val) {
    if (val) return "true";
    return "false";
  }

};

namespace XtcInput {

std::string StreamDgram::streamType2str(const StreamDgram::StreamType val) {
  switch (val) {
  case StreamDgram::DAQ:
    return "DAQ";
  case StreamDgram::controlUnderDAQ:
    return "controlUnderDAQ";
  case StreamDgram::controlIndependent:
    return "controlIndependent";
  }
    return "*unknown*";
}

std::string StreamDgram::dumpStr(const StreamDgram & dg) {
  std::ostringstream msg;
  if (dg.empty()) return "empty dgram";
  const Pds::Dgram *dgram = dg.dg().get();
  const Pds::Sequence & seq = dgram->seq;
  const Pds::ClockTime & clock = seq.clock();
  const Pds::TimeStamp & stamp = seq.stamp();
  msg << streamType2str(dg.streamType())
      << " streamId=" << dg.streamId()
      << " L1Block=" << dg.L1Block()
      << " sv=" << Pds::TransitionId::name(seq.service())
      << " ev=" << seq.isEvent()
      << " sec=0x" << std::hex << std::setw(9) << clock.seconds()
      << " nano=0x" << std::hex << std::setw(9) << clock.nanoseconds()
      << " fid=0x" << std::hex << std::setw(7) << stamp.fiducials()
      << " ctrl=" << std::dec << stamp.control()
      << " vec=" << std::dec << stamp.vector()
      << " streamNo=" << std::dec << std::setw(2) << dg.file().stream()
      << " file=" << dg.file().path();
  return msg.str();
}

StreamDgramGreater::StreamDgramGreater(unsigned maxClockDriftSeconds) 
  : m_fidCompare(maxClockDriftSeconds) {
  DgramCategory LD(L1Accept, StreamDgram::DAQ);
  DgramCategory LC(L1Accept, StreamDgram::controlUnderDAQ);
  DgramCategory LI(L1Accept, StreamDgram::controlIndependent);
  DgramCategory TD(otherTrans, StreamDgram::DAQ);
  DgramCategory TC(otherTrans, StreamDgram::controlUnderDAQ);
  DgramCategory TI(otherTrans, StreamDgram::controlIndependent);

  /* -------------------------------------------------------------------
     Below we encode the 21 cases for comparing the 6 Dgram catagorires (LD, LC, LI, TD, TC, TI) 
     defined above, against one another (there are 36 pairs from these 6 categories, but the
     compare method does not depend on the order of the pair: TD vs TI is the same as TI vs TD.

     The 21 cases cover all combinations we may see when merging dgrams from streams. 
     Issues that go into the merging rules:
     *  There may be multiple C streams (s80, s81)
     *  Not all L1Accepts in a C stream will have a matching L1Accept in the DAQ stream. 
     *  We do not want to throw away L1Accepts from the s80, if they do not match a 
        DAQ stream, we want to order them correctly.
     *  Comparing a C stream L1 accept against D or C stream Transitions requires history.
        The clocks are different and fiducials in both are not available. The C stream
        L1 accept need not have a matching L1 in the Daq stream. This is when the block number 
        is used.
     *  Comparing a DAQ L1 accept against a DAQ Transition also requires history. We are not
        guaranteed that the clock for an L1 accept following a DAQ transition has a later time
        than the clock for the transition. This is most always the case, but there has been 
        data where it is not true. We assume that all L1 accepts before the transition 
        should appear earlier, and likewise all L1 accepts after should appear after. The
        block is used in these cases.
     *  At this point, the C and I streams are not supposed to have anything useful in their
        Transition dgrams (other than the Configure transition). 
     *  We cannot compare anything from independent streams. One the one hand, a LD vs. an LI is
        fiducials based, but we first check the run and L1Block count. Before we compare anything
        from an independent stream, we need a process that assigns L1Block numbers to Dgrams from
        the Independent streams that are synchronized with those of the DAQ/Control streams. 
        Hence all comparisions involving a TI or LI (even against themselves) are presently marked
        badGreater.
     ------------------------------------------------------------------- */
  m_LUT[makeDgramCategoryAB(LD,LD)] = fidGreater;
  m_LUT[makeDgramCategoryAB(LD,LC)] = fidGreater;
  m_LUT[makeDgramCategoryAB(LD,LI)] = badGreater;
  m_LUT[makeDgramCategoryAB(LD,TD)] = blockGreater;
  m_LUT[makeDgramCategoryAB(LD,TC)] = blockGreater;
  m_LUT[makeDgramCategoryAB(LD,TI)] = badGreater; 

  m_LUT[makeDgramCategoryAB(LC,LC)] = fidGreater;
  m_LUT[makeDgramCategoryAB(LC,LI)] = badGreater;
  m_LUT[makeDgramCategoryAB(LC,TD)] = blockGreater;
  m_LUT[makeDgramCategoryAB(LC,TC)] = blockGreater;
  m_LUT[makeDgramCategoryAB(LC,TI)] = badGreater; 

  m_LUT[makeDgramCategoryAB(LI,LI)] = badGreater;
  m_LUT[makeDgramCategoryAB(LI,TD)] = badGreater;
  m_LUT[makeDgramCategoryAB(LI,TC)] = badGreater;
  m_LUT[makeDgramCategoryAB(LI,TI)] = badGreater;

  m_LUT[makeDgramCategoryAB(TD,TD)] = clockGreater;
  m_LUT[makeDgramCategoryAB(TD,TC)] = clockGreater;
  m_LUT[makeDgramCategoryAB(TD,TI)] = badGreater;

  m_LUT[makeDgramCategoryAB(TC,TC)] = clockGreater;
  m_LUT[makeDgramCategoryAB(TC,TI)] = badGreater;

  m_LUT[makeDgramCategoryAB(TI,TI)] = badGreater;
}
  
StreamDgramGreater::DgramCategory StreamDgramGreater::getDgramCategory(const StreamDgram &dg) {
  if (dg.empty()) {
    MsgLog(logger, warning, "getDgramCategory called on empty dgram");
    return StreamDgramGreater::DgramCategory(L1Accept, StreamDgram::DAQ);
  }

  TransitionType trans;
  if (dg.dg()->seq.service() == Pds::TransitionId::L1Accept) {
    trans = L1Accept;
  } else {
    trans = otherTrans;
  }
  return StreamDgramGreater::DgramCategory(trans, dg.streamType());
}

StreamDgramGreater::DgramCategoryAB StreamDgramGreater::makeDgramCategoryAB(DgramCategory a, DgramCategory b) {
  return StreamDgramGreater::DgramCategoryAB(a,b);
}

// implement greater than, 
bool StreamDgramGreater::operator()(const StreamDgram &a, const StreamDgram &b) const {
  // two empty datagrams are equal to one another
  if (a.empty() and b.empty()) return false;
 
  // an empty dgram is always greater than a non-empty one, empty dgrams should 
  // appear last in an ordered list of dgrams
  if (a.empty()) return true;
  if (b.empty()) return false;

  StreamDgramGreater::DgramCategory dgramCategA = getDgramCategory(a);
  StreamDgramGreater::DgramCategory dgramCategB = getDgramCategory(b);
  StreamDgramGreater::DgramCategoryAB dgramCategAB = makeDgramCategoryAB(dgramCategA, dgramCategB);
  std::map<DgramCategoryAB, CompareMethod>::const_iterator pos = m_LUT.find(dgramCategAB);
  if (pos == m_LUT.end()) {
    StreamDgramGreater::DgramCategoryAB dgramCategBA = makeDgramCategoryAB(dgramCategB, dgramCategA);
    pos = m_LUT.find(dgramCategBA);
    if (pos == m_LUT.end()) throw UnknownGreater(ERR_LOC);
  }
  StreamDgramGreater::CompareMethod compareMethod = pos->second;
  
  switch (compareMethod) {
  case clockGreater:
    return doClockGreater(a,b);
  case fidGreater:
    return doFidGreater(a,b);
  case blockGreater:
    return doBlockGreater(a,b);
  case badGreater:
    return doBadGreater(a,b);
  }

  MsgLog(logger, fatal, "StreamDgramGreater: unexpected error. compare method in look up table = " 
         << int(compareMethod) << " was not handled in switch statement");
  return false;
}

bool StreamDgramGreater::doClockGreater(const StreamDgram &a, const StreamDgram &b) const
{ 
  if (a.empty() or b.empty()) throw psana::Exception(ERR_LOC, "StreamDgramGreater: empty dgs");
  int runResult = runLessGreater(a,b);
  if (runResult > 0) return true;
  if (runResult < 0) return false;
  int blockResult = blockLessGreater(a,b);
  if (blockResult > 0) return true;
  if (blockResult < 0) return false;
  const Pds::ClockTime & clockA = a.dg()->seq.clock();
  const Pds::ClockTime & clockB = b.dg()->seq.clock();
  bool res = clockA > clockB;
  return res;
}

bool StreamDgramGreater::doFidGreater(const StreamDgram &a, const StreamDgram &b) const
{ 
  if (a.empty() or b.empty()) throw psana::Exception(ERR_LOC, "StreamDgramGreater: empty dgs");
  int runResult = runLessGreater(a,b);
  if (runResult > 0) return true;
  if (runResult < 0) return false;
  int blockResult = blockLessGreater(a,b);
  if (blockResult > 0) return true;
  if (blockResult < 0) return false;
  bool res = m_fidCompare.fiducialsGreater(*a.dg(), *b.dg());
  return res;
}

bool StreamDgramGreater::doBlockGreater(const StreamDgram &a, const StreamDgram &b) const
{ 
  if (a.empty() or b.empty()) throw psana::Exception(ERR_LOC, "StreamDgramGreater: empty dgs");

  TransitionType transA = getDgramCategory(a).first;
  TransitionType transB = getDgramCategory(b).first;

  if (((transA == L1Accept) and (transB == L1Accept)) or
      ((transA == otherTrans) and (transB == otherTrans))) {
    throw psana::Exception(ERR_LOC, "DoBlockGreater: both datagrams are "
                           "either L1Accept or otherTrans. They must be mixed");
  }

  // first compare runs. The block number should be reset 0 when processing a new run
  // in the same stream. One could imagine implement a running block count accross all the runs
  // for a stream, but if a run is omitted in one stream and not the others, a running block 
  // number across all the runs would get out of sync. Also sometimes runs are brought to an
  // end because of DAQ problems with synchronizing the non L1Accept transitions accross the
  // streams - so a block count they may not be synchronized accross streams at the very end of a run.

  // compare runs
  unsigned runA = a.file().run();
  unsigned runB = b.file().run();

  if (runA < runB) return false;
  if (runA > runB) return true;

  // same run, compare block number.

  if ((transA == L1Accept) and (transB == otherTrans)) {
    bool res = a.L1Block() >= b.L1Block();
    return res;
  } 
  // (transA == otherTrans) and (transB == L1Accept)
  bool res = a.L1Block() > b.L1Block();
  return res;
}

bool StreamDgramGreater::doBadGreater(const StreamDgram &a, const StreamDgram &b) const
{ 
  throw psana::Exception(ERR_LOC, "doBadGreater called");
}

int StreamDgramGreater::runLessGreater(const StreamDgram &a, const StreamDgram &b) const 
{
  unsigned runA = a.file().run();
  unsigned runB = b.file().run();
  if (runA < runB) return -1;
  if (runA > runB) return 1;
  return 0;
}

int StreamDgramGreater::blockLessGreater(const StreamDgram &a, const StreamDgram &b) const
{
  int64_t blockA = a.L1Block();
  int64_t blockB = b.L1Block();

  if (blockA < blockB) return -1;
  if (blockA > blockB) return 1;
  return 0;
}

bool StreamDgramGreater::sameEvent(const StreamDgram &a, const StreamDgram &b) const {
  if (a.empty() and b.empty()) {
    MsgLog(logger, warning, "sameEvent: comparing two empty dgrams");
    return true;
  }
  if  (a.empty() or b.empty()) {
    MsgLog(logger, warning, "sameEvent: comparing an empty dgram to a non-empty dgram");
    return false;
  }
  throw psana::Exception(ERR_LOC, "sameEvent: not implmented");
}
  
};
