//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class DgramSourceFile...
//
// Author List:
//     Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcInput/DgramSourceFile.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSXtcInput/Exceptions.h"
#include "XtcInput/DgramQueue.h"
#include "XtcInput/DgramReader.h"
#include "XtcInput/MergeMode.h"
#include "XtcInput/XtcFilesPosition.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace XtcInput;

//             ----------------------------------------
//             -- Public Function Member Definitions --
//             ----------------------------------------

namespace {

  const unsigned MAX_SEC_DRIFT_FOR_FIDUCIAL_MATCH = 90;

  bool isL1Accept(const XtcInput::Dgram &dg) {
    if (dg.empty()) return false;
    XtcInput::Dgram::ptr pDg = dg.dg();
    return pDg->seq.service() == Pds::TransitionId::L1Accept;
  }

  bool transitionsMatch(const XtcInput::Dgram &dgA, const XtcInput::Dgram &dgB) {
    if (dgA.empty() or dgB.empty()) return false;
    XtcInput::Dgram::ptr pDgA = dgA.dg();
    XtcInput::Dgram::ptr pDgB = dgB.dg();
    return pDgA->seq.service() == pDgB->seq.service();
  }

  bool isFiducialMatchStream(const XtcInput::Dgram &dg, const int firstControlStream) {
    if (dg.empty()) return false;
    const XtcFileName& fileName = dg.file();
    if (fileName.empty()) return false;
    return (int(fileName.stream()) >= firstControlStream);
  }
  
  unsigned absDiff(unsigned a, unsigned b) {
    if (b >= a) return b-a;
    return a-b;
  }

  bool clockTimesMatch(const XtcInput::Dgram &dgA, const XtcInput::Dgram &dgB) {
    if (dgA.empty() or dgB.empty()) return false;
    XtcInput::Dgram::ptr pDgA = dgA.dg();
    XtcInput::Dgram::ptr pDgB = dgB.dg();
    unsigned secA = pDgA->seq.clock().seconds();
    unsigned secB = pDgB->seq.clock().seconds();
    if (secA != secB) return false;
    unsigned nanoA = pDgA->seq.clock().nanoseconds();
    unsigned nanoB = pDgB->seq.clock().nanoseconds();
    return nanoA == nanoB;
  }

}

namespace PSXtcInput {

//----------------
// Constructors --
//----------------
DgramSourceFile::DgramSourceFile (const std::string& name)
  : IDatagramSource()
  , psana::Configurable(name)
  , m_dgQueue(new XtcInput::DgramQueue(10))
  , m_readerThread()
  , m_fileNames()
  , m_firstControlStream(80)

{
  m_fileNames = configList("files");
  if ( m_fileNames.empty() ) {
    throw EmptyFileList(ERR_LOC);
  }
}

//--------------
// Destructor --
//--------------
DgramSourceFile::~DgramSourceFile ()
{
  if (m_readerThread) {
    // ask the thread to stop
    m_readerThread->interrupt();
    MsgLog(name(), debug, "wait for reader thread to finish");
    // wait until it does
    m_readerThread->join();
    MsgLog(name(), debug, "reader thread has finished");
  }
}

// Initialization method for datagram source
void 
DgramSourceFile::init()
{
  // will throw if no files were defined in config
  WithMsgLog(name(), debug, str) {
    str << "Input files: ";
    std::copy(m_fileNames.begin(), m_fileNames.end(), std::ostream_iterator<std::string>(str, " "));
  }
  
  // start reader thread
  std::string liveDbConn = configStr("liveDbConn", "");
  std::string liveTable = configStr("liveTable", "file");
  unsigned liveTimeout = config("liveTimeout", 120U);
  unsigned runLiveTimeout = config("runLiveTimeout", 0U);
  double l1offset = config("l1offset", 0.0);
  MergeMode merge = mergeMode(configStr("mergeMode", "FileName"));
  m_firstControlStream = config("first_control_stream",80);
  m_maxStreamClockDiffSec = config("max_stream_clock_diff",85);

  std::list<off64_t> emptyOffsets;
  std::list<off64_t> offsets =  configList("third_event_jump_offsets",emptyOffsets);
  std::list<std::string> emptyStrings;
  std::list<std::string> filenames = configList("third_event_jump_filenames",emptyStrings);

  boost::shared_ptr<XtcFilesPosition> firstEventAfterConfigure;
  if ((offsets.size() > 0) or (filenames.size() > 0)) {
    firstEventAfterConfigure = boost::make_shared<XtcFilesPosition>(filenames,
                                                                    offsets);
  }
  m_readerThread.reset( new boost::thread( DgramReader ( m_fileNames.begin(), 
                                                         m_fileNames.end(),
                                                         *m_dgQueue, 
                                                         merge, liveDbConn, 
                                                         liveTable, liveTimeout, runLiveTimeout,
                                                         l1offset, 
                                                         m_firstControlStream,
                                                         m_maxStreamClockDiffSec,
                                                         firstEventAfterConfigure) ) );
  MsgLog(name(), debug, "config params: liveDbConn=" << liveDbConn << ", " 
         << "liveTable=" << liveTable << ", "
         << "liveTimeout=" << liveTimeout << ", "
         << "runLiveTimeout=" << runLiveTimeout << ", "
         << "l1offset=" << l1offset << ", "
         << "mergeMode=" << merge << ", "
         << "first_control_stream=" << m_firstControlStream << ","
         << "max_stream_clock_diff=" << m_maxStreamClockDiffSec);
}


//  This method returns next datagram from the source
bool
DgramSourceFile::next(std::vector<XtcInput::Dgram>& eventDg, std::vector<XtcInput::Dgram>& nonEventDg)
{
  XtcInput::Dgram dg = m_dgQueue->pop();
  if (not dg.empty()) {
    eventDg.push_back(dg);
    bool foundDgramForDifferentEvent = false;
    while (not foundDgramForDifferentEvent) {
      XtcInput::Dgram nextDg =  m_dgQueue->front();
      if (sameEvent(dg, nextDg)) {
        nextDg = m_dgQueue->pop();
        eventDg.push_back(nextDg);
      } else {
        foundDgramForDifferentEvent = true;
      }
    }
    return true;
  } else {
    return false;
  }
}

bool DgramSourceFile::sameEvent(const XtcInput::Dgram &eventDg, const XtcInput::Dgram &otherDg) const
{
  if (::isL1Accept(otherDg) and ::isL1Accept(eventDg) and 
      (::isFiducialMatchStream(otherDg, m_firstControlStream) or
       ::isFiducialMatchStream(eventDg, m_firstControlStream)) and
      fiducialSecondsMatch(eventDg, otherDg)) {
    return true;
  }
  if ((not ::isL1Accept(otherDg)) and
      (not ::isL1Accept(eventDg)) and 
      ::transitionsMatch(otherDg, eventDg) and
      ::clockTimesMatch(otherDg, eventDg)) {
    return true;
  }
  return false;
}

bool DgramSourceFile::fiducialSecondsMatch(const XtcInput::Dgram &dgA, const XtcInput::Dgram &dgB) const {
  if (dgA.empty() or dgB.empty()) return false;
  XtcInput::Dgram::ptr pDgA = dgA.dg();
  XtcInput::Dgram::ptr pDgB = dgB.dg();
  unsigned fidA = pDgA->seq.stamp().fiducials();
  unsigned fidB = pDgB->seq.stamp().fiducials();
  if (fidA != fidB) return false;
  unsigned secA = pDgA->seq.clock().seconds();
  unsigned secB = pDgB->seq.clock().seconds();
  unsigned drift = ::absDiff(secA,secB);
  return drift < m_maxStreamClockDiffSec;
}


} // namespace PSXtcInput
