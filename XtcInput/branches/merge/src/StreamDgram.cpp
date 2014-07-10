#include <iomanip>

#include "MsgLogger/MsgLogger.h"
#include "XtcInput/StreamDgram.h"

#define DVDMSG debug

namespace {
  char *logger = "StreamDgram";

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
  //  const Pds::Env & env = dgram->env;
  const Pds::ClockTime & clock = seq.clock();
  const Pds::TimeStamp & stamp = seq.stamp();
  msg << streamType2str(dg.streamType())
      << " streamId=" << dg.streamId()
      << " L1Block=" << dg.L1Block()
    //      << " tp=" << int(seq.type())
      << " sv=" << Pds::TransitionId::name(seq.service())
    //      << " ex=" << seq.isExtended()
      << " ev=" << seq.isEvent()
      << " sec=" << std::hex << std::setw(9) << clock.seconds()
      << " nano=" << std::hex << std::setw(9) << clock.nanoseconds()
    //      << " tcks=" << std::hex << std::setw(12) << stamp.ticks()
      << " fid=" << std::hex << std::setw(7) << stamp.fiducials()
      << " ctrl=" << stamp.control()
      << " vec=" << stamp.vector()
    //      << " env=" << env.value()
      << " streamNo=" << std::setw(2) << dg.file().stream()
      << " file=" << dg.file().path();
  return msg.str();
}

StreamDgramCmp::StreamDgramCmp(const boost::shared_ptr<ExperimentClockDiffMap> expClockDiff, 
                                 unsigned maxClockDriftSeconds) 
  : m_expClockDiff(expClockDiff), m_fidCompare(maxClockDriftSeconds) {
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
     *  Not all L1 accepts in a C stream will have a matching L1 Accept in the DAQ stream. 
        We want to order them properly, the consumer class can decide if it wants to use 
        non-matching C stream L1 accepts.
     *  Comparing a C stream L1 accept against D or C stream Transitions requires history.
        The clocks are different and fiducials in both are not available, and the C stream
        L1 accept need not have a matching L1 in the Daq stream. This is when the block number 
        is used.
     *  Comparing a DAQ L1 accept against a DAQ Transition also requires history. We are not
        guaranteed that the clock for an L1 accept following a DAQ transition has a later time
        than the clock for the transition. We assume that all L1 accepts before the transition 
        should appear earlier, and likewise all L1 accepts after should appear after.
      
     *  Comparing from a D to a I stream is difficult. The clocks are different, and the 
        Transitions are not synchronized. A fiducial compare is reasonable for L1 vs. L1, but 
        to compare Transitions between D and I, we are stuck. If we knew the difference in the 
        clocks down to the nanosecond we could determine when two transitions are equal, but 
        also, how is that useful? If we compare a Transition in a DAQ stream against a L1 accept
        in a Independent stream, we are more stuck. We can't rely on clocks to compare T vs L1,
        and comparing block numbers doesn't make sense between I and D. We call all these badCmp
    
     ------------------------------------------------------------------- */
  m_LUT[makeDgramCategoryAB(LD,LD)] = clockCmp;
  m_LUT[makeDgramCategoryAB(LD,LC)] = fidCmp;
  m_LUT[makeDgramCategoryAB(LD,LI)] = fidCmp;
  m_LUT[makeDgramCategoryAB(LD,TD)] = blockCmp;
  m_LUT[makeDgramCategoryAB(LD,TC)] = blockCmp;
  m_LUT[makeDgramCategoryAB(LD,TI)] = badCmp; 

  m_LUT[makeDgramCategoryAB(LC,LC)] = fidCmp;
  m_LUT[makeDgramCategoryAB(LC,LI)] = fidCmp;
  m_LUT[makeDgramCategoryAB(LC,TD)] = blockCmp;
  m_LUT[makeDgramCategoryAB(LC,TC)] = blockCmp;
  m_LUT[makeDgramCategoryAB(LC,TI)] = badCmp; 

  m_LUT[makeDgramCategoryAB(LI,LI)] = clockCmp;
  m_LUT[makeDgramCategoryAB(LI,TD)] = badCmp;
  m_LUT[makeDgramCategoryAB(LI,TC)] = badCmp;
  m_LUT[makeDgramCategoryAB(LI,TI)] = blockCmp;

  m_LUT[makeDgramCategoryAB(TD,TD)] = clockCmp;
  m_LUT[makeDgramCategoryAB(TD,TC)] = clockCmp;
  m_LUT[makeDgramCategoryAB(TD,TI)] = mapCmp;

  m_LUT[makeDgramCategoryAB(TC,TC)] = clockCmp;
  m_LUT[makeDgramCategoryAB(TC,TI)] = mapCmp;

  m_LUT[makeDgramCategoryAB(TI,TI)] = clockCmp;
}
  
StreamDgramCmp::DgramCategory StreamDgramCmp::getDgramCategory(const StreamDgram &dg) {
  if (dg.empty()) {
    MsgLog(logger, warning, "getDgramCategory called on empty dgram");
    return StreamDgramCmp::DgramCategory(L1Accept, StreamDgram::DAQ);
  }

  TransitionType trans;
  if (dg.dg()->seq.service() == Pds::TransitionId::L1Accept) {
    trans = L1Accept;
  } else {
    trans = otherTrans;
  }
  return StreamDgramCmp::DgramCategory(trans, dg.streamType());
}

StreamDgramCmp::DgramCategoryAB StreamDgramCmp::makeDgramCategoryAB(DgramCategory a, DgramCategory b) {
  return StreamDgramCmp::DgramCategoryAB(a,b);
}

// implement greater than, 
bool StreamDgramCmp::operator()(const StreamDgram &a, const StreamDgram &b) const {
  // two empty datagrams are equal to one another
  if (a.empty() and b.empty()) return false;
 
  // an empty dgram is always greater than a non-empty one, empty dgrams should 
  // appear last in an ordered list of dgrams
  if (a.empty()) return true;
  if (b.empty()) return false;

  StreamDgramCmp::DgramCategory dgramCategA = getDgramCategory(a);
  StreamDgramCmp::DgramCategory dgramCategB = getDgramCategory(b);
  StreamDgramCmp::DgramCategoryAB dgramCategAB = makeDgramCategoryAB(dgramCategA, dgramCategB);
  std::map<DgramCategoryAB, CompareMethod>::const_iterator pos = m_LUT.find(dgramCategAB);
  if (pos == m_LUT.end()) {
    StreamDgramCmp::DgramCategoryAB dgramCategBA = makeDgramCategoryAB(dgramCategB, dgramCategA);
    pos = m_LUT.find(dgramCategBA);
    if (pos == m_LUT.end()) throw UnknownCmp(ERR_LOC);
  }
  StreamDgramCmp::CompareMethod compareMethod = pos->second;
  
  switch (compareMethod) {
  case clockCmp:
    return doClockCmp(a,b);
  case fidCmp:
    return doFidCmp(a,b);
  case blockCmp:
    return doBlockCmp(a,b);
  case mapCmp:
    return doMapCmp(a,b);
  case badCmp:
    return doBadCmp(a,b);
  }

  MsgLog(logger, fatal, "StreamDgramCmp: unexpected error. compare method in look up table = " 
         << int(compareMethod) << " was not handled in switch statement");
  return false;
}

// return true is a > b
bool StreamDgramCmp::doClockCmp(const StreamDgram &a, const StreamDgram &b) const
{ 
  if (a.empty() or b.empty()) throw psana::Exception(ERR_LOC, "StreamDgramCmp: empty dgs");
  int runResult = runLessGreater(a,b);
  if (runResult > 0) return true;
  if (runResult < 0) return false;
  int blockResult = blockLessGreater(a,b);
  if (blockResult > 0) return true;
  if (blockResult < 0) return false;
  const Pds::ClockTime & clockA = a.dg()->seq.clock();
  const Pds::ClockTime & clockB = b.dg()->seq.clock();
  bool res = clockA > clockB;
  MsgLog(logger,DVDMSG, "doClockCmp: A > B is " << bool2str(res) << " dgrams: " << std::endl 
         << "A: " << StreamDgram::dumpStr(a) << std::endl
         << "B: " << StreamDgram::dumpStr(b));
  return res;
}

// return true is a > b
bool StreamDgramCmp::doFidCmp(const StreamDgram &a, const StreamDgram &b) const
{ 
  if (a.empty() or b.empty()) throw psana::Exception(ERR_LOC, "StreamDgramCmp: empty dgs");
  int runResult = runLessGreater(a,b);
  if (runResult > 0) return true;
  if (runResult < 0) return false;
  int blockResult = blockLessGreater(a,b);
  if (blockResult > 0) return true;
  if (blockResult < 0) return false;
  bool res = m_fidCompare.fiducialsGreater(*a.dg(), *b.dg());
  MsgLog(logger,DVDMSG, "doFidCmp: A > B is " << bool2str(res) << " dgrams: " << std::endl 
         << "A: " << StreamDgram::dumpStr(a) << std::endl
         << "B: " << StreamDgram::dumpStr(b));
  
  return res;
}

bool StreamDgramCmp::doBlockCmp(const StreamDgram &a, const StreamDgram &b) const
{ 
  if (a.empty() or b.empty()) throw psana::Exception(ERR_LOC, "StreamDgramCmp: empty dgs");

  TransitionType transA = getDgramCategory(a).first;
  TransitionType transB = getDgramCategory(b).first;

  if (((transA == L1Accept) and (transB == L1Accept)) or
      ((transA == otherTrans) and (transB == otherTrans))) {
    throw psana::Exception(ERR_LOC, "DoBlockCmp: both datagrams are "
                           "either L1Accept or otherTrans. They must be mixed");
  }

  // first compare runs. The block number should be reset 0 when processing a new run
  // in the same stream. One could imagine implement a running block count accross runs
  // but if a run omitted one stream, a running block number across all
  // the runs would get out of sync (accross different streams).

  // compare runs
  unsigned runA = a.file().run();
  unsigned runB = b.file().run();

  double secondsA = a.dg()->seq.clock().asDouble();
  double secondsB = b.dg()->seq.clock().asDouble();
  double AminusB = secondsA - secondsB;

  if (runA < runB) {
    if (AminusB > m_fidCompare.maxClockDriftSeconds()) {
      MsgLog(logger, warning, "doBlockCmp: dgram A is in earler run but clock is more than "
             << m_fidCompare.maxClockDriftSeconds() << " seconds later than dgram B");
    }
    MsgLog(logger, DVDMSG, "doBlockCmp: A > B = false as runA=" << runA << " runB=" << runB 
           << " dgrams:" << std::endl
           << "A: " << StreamDgram::dumpStr(a) << std::endl
           << "B: " << StreamDgram::dumpStr(b));
    return false;
  }
  if (runA > runB) {
    if (AminusB < (-1.0*double(m_fidCompare.maxClockDriftSeconds()))) {
      MsgLog(logger, warning, "doBlockCmp: dgram A is in later run but clock is more than "
             << m_fidCompare.maxClockDriftSeconds() << " seconds earlier than dgram B");
    }
    MsgLog(logger, DVDMSG, "doBlockCmp: A > B = true as runA=" << runA << " runB=" << runB 
           << " dgrams:" << std::endl
           << "A: " << StreamDgram::dumpStr(a) << std::endl
           << "B: " << StreamDgram::dumpStr(b));
    return true;
  }

  // same run, compare block number.

  if ((transA == L1Accept) and (transB == otherTrans)) {
    bool res = a.L1Block() >= b.L1Block();
    MsgLog(logger, DVDMSG, "doBlockCmp: same run. A > B = " << bool2str(res)
           << " dgrams:" << std::endl
           << "A: " << StreamDgram::dumpStr(a) << std::endl
           << "B: " << StreamDgram::dumpStr(b));
    return res;
  } 
  // (transA == otherTrans) and (transB == L1Accept)
  bool res = a.L1Block() > b.L1Block();
  MsgLog(logger, DVDMSG, "doBlockCmp: same run. A > B = " << bool2str(res)
         << " dgrams:" << std::endl
         << "A: " << StreamDgram::dumpStr(a) << std::endl
         << "B: " << StreamDgram::dumpStr(b));
  return res;
}

bool StreamDgramCmp::doMapCmp(const StreamDgram &a, const StreamDgram &b) const
{ 
  if (a.empty() or b.empty()) throw psana::Exception(ERR_LOC, "StreamDgramCmp: empty dgs");
  if (not m_expClockDiff) throw psana::Exception(ERR_LOC, "doMapCmp: expClockDiff map is null");
  unsigned expA = a.file().expNum();
  unsigned expB = b.file().expNum();
  if ((expA == 0) or (expB == 0)) throw psana::Exception(ERR_LOC, "doMapCmp: an experiment number is 0");
  ExperimentPair experimentsAB(expA, expB);
  bool abInMap = true;
  ExperimentClockDiffMap::const_iterator pos = m_expClockDiff->find(experimentsAB);
  if (pos == m_expClockDiff->end()) {
    ExperimentPair experimentsBA(expB, expA);
    pos = m_expClockDiff->find(experimentsBA);
    if (pos == m_expClockDiff->end()) {
      throw NoClockDiff(ERR_LOC, expA, expB);
    }
    abInMap = false;
  }
  // now we need to add the clockDiff found in the map to a's clockTime (if abInMap is true)
  // or add it to the clockTime for b's clockTime (if abInMap is false) and do the 
  // normal clock comparison
  throw psana::Exception(ERR_LOC, "doMapCmp: not implmented");
}

bool StreamDgramCmp::doBadCmp(const StreamDgram &a, const StreamDgram &b) const
{ 
  throw psana::Exception(ERR_LOC, "doBadCmp called");
}

int StreamDgramCmp::runLessGreater(const StreamDgram &a, const StreamDgram &b) const 
{
  unsigned runA = a.file().run();
  unsigned runB = b.file().run();
  if (runA < runB) return -1;
  if (runA > runB) return 1;
  return 0;
}

int StreamDgramCmp::blockLessGreater(const StreamDgram &a, const StreamDgram &b) const
{
  int64_t blockA = a.L1Block();
  int64_t blockB = b.L1Block();

  if (blockA < blockB) return -1;
  if (blockA > blockB) return 1;
  return 0;
}

bool StreamDgramCmp::sameEvent(const StreamDgram &a, const StreamDgram &b) const {
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
