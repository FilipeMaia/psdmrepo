//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ShmemMonitorClient...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSShmemInput/ShmemMonitorClient.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/thread.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/Dgram.hh"
#include "XtcInput/Dgram.h"
#include "XtcInput/DgramQueue.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "PSShmemInput.ShmemMonitorClient";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSShmemInput {

//----------------
// Constructors --
//----------------
ShmemMonitorClient::ShmemMonitorClient (const std::string& tag, int index,
    XtcInput::DgramQueue& queue, Pds::TransitionId::Value stopTr)
  : Pds::XtcMonitorClient()
  , m_tag(tag)
  , m_index(index)
  , m_queue(queue)
  , m_stopTr(stopTr)
{
}

// this is the "run" method used by the Boost.thread
void
ShmemMonitorClient::operator()()
{
  this->run(m_tag.c_str(), m_index);
}

// overriding base class method
int
ShmemMonitorClient::processDgram(Pds::Dgram* dg)
{
  Pds::TransitionId::Value tr = dg->seq.service();
  MsgLog(logger, debug, "received transition " << Pds::TransitionId::name(tr));

  // skip some datagrams that base class does not expect
  if (tr != Pds::TransitionId::Map and tr != Pds::TransitionId::Unmap) {
    // copy datagram
    const unsigned size = sizeof(Pds::Dgram) + dg->xtc.sizeofPayload();
    const char* src = (const char*)dg;
    char* dst = new char[size];
    std::copy(src, src+size, dst);

    // wrap into object and push into queue
    // need to create a large run number so shmem analysis can pick
    // up the latest calibration constants.
    XtcInput::Dgram dgobj(XtcInput::Dgram::make_ptr((Pds::Dgram*)dst), XtcInput::XtcFileName("<ShMem>",0,std::numeric_limits<short>::max()-1,0,0));
    m_queue.push(dgobj);
  }

  // check if we should stop now
  bool stop = false;
  switch (tr) {
  case Pds::TransitionId::Unmap:
    stop = m_stopTr == Pds::TransitionId::Unmap or
           m_stopTr == Pds::TransitionId::Unconfigure or
           m_stopTr == Pds::TransitionId::EndRun or
           m_stopTr == Pds::TransitionId::EndCalibCycle;
    break;
  case Pds::TransitionId::Unconfigure:
    stop = m_stopTr == Pds::TransitionId::Unconfigure or
           m_stopTr == Pds::TransitionId::EndRun or
           m_stopTr == Pds::TransitionId::EndCalibCycle;
    break;
  case Pds::TransitionId::EndRun:
    stop = m_stopTr == Pds::TransitionId::EndRun or
           m_stopTr == Pds::TransitionId::EndCalibCycle;
    break;
  case Pds::TransitionId::EndCalibCycle:
    stop = m_stopTr == Pds::TransitionId::EndCalibCycle;
    break;
  default:
    break;
  }

  // if signalled try to stop gracefully
  if (stop) {
    MsgLog(logger, debug, "stop transition reached");
    m_queue.push(XtcInput::Dgram());
    return 1;
  } else if (boost::this_thread::interruption_requested()) {
    MsgLog(logger, debug, "interrupted, stopping");
    m_queue.push(XtcInput::Dgram());
    return -1;
  } else {
    return 0;
  }
}

} // namespace PSShmemInput
