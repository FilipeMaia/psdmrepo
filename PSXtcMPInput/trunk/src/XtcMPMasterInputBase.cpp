//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcMPMasterInputBase...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcMPInput/XtcMPMasterInputBase.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <vector>
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/L1AcceptEnv.hh"
#include "PSXtcMPInput/Exceptions.h"
#include "PSXtcMPInput/XtcMPDgramSerializer.h"
#include "XtcInput/XtcIterator.h"
#include "XtcInput/XtcFilter.h"
#include "XtcInput/XtcFilterTypeIdSrc.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace XtcInput;

namespace {


  // return true if container contains EPICS data only
  bool epicsOnly(Pds::Xtc* xtc)
  {
    XtcInput::XtcIterator iter(xtc);
    while (Pds::Xtc* x = iter.next()) {
      switch (x->contains.id()) {
      case Pds::TypeId::Id_Xtc:
      case Pds::TypeId::Id_Epics:
        continue;
      default:
        return false;
      }
    }
    return true;
  }

  // return true if event passed L3 selection (of there was no L3 defined)
  bool l3accept(const Pds::Dgram& dg)
  {
    return not static_cast<const Pds::L1AcceptEnv&>(dg.env).trimmed();
  }

  // special delete for boost::shared_ptr
  struct CharArrayDeleter {
    void operator()(char* p) { delete [] p; }
  };

}



//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcMPInput {

//----------------
// Constructors --
//----------------
XtcMPMasterInputBase::XtcMPMasterInputBase (const std::string& name, const boost::shared_ptr<PSXtcInput::IDatagramSource>& dgsource)
  : InputModule(name)
  , m_dgsource(dgsource)
  , m_putBack()
  , m_skipEvents(0)
  , m_maxEvents(0)
  , m_skipEpics(false)
  , m_l3tAcceptOnly(true)
  , m_l1Count(0)
  , m_epicsDgs()
  , m_workers()
{
  std::fill_n(m_transitions, int(Pds::TransitionId::NumberOf), Pds::ClockTime(0, 0));

  // get number of events to process/skip from psana configuration
  ConfigSvc::ConfigSvc cfg = configSvc();
  m_skipEvents = cfg.get("psana", "skip-events", 0UL);
  m_maxEvents = cfg.get("psana", "events", 0UL);
  m_skipEpics = cfg.get("psana", "skip-epics", true);
  m_l3tAcceptOnly = cfg.get("psana", "l3t-accept-only", true);

  m_fdReadyPipe = config("fdReadyPipe");
}

//--------------
// Destructor --
//--------------
XtcMPMasterInputBase::~XtcMPMasterInputBase ()
{
}

/// Method which is called once at the beginning of the job
void 
XtcMPMasterInputBase::beginJob(Event& evt, Env& env)
{
  MsgLog(name(), debug, name() << ": in beginJob()");

  // get worker info and store it locally
  boost::shared_ptr<std::vector<MPWorkerId> > workers = env.configStore().get(Pds::Src());;
  if (not workers) {
    throw MissingWorkers(ERR_LOC);
  }
  BOOST_FOREACH (const MPWorkerId& worker, *workers) {
    m_workers.insert(std::make_pair(worker.workerId(), worker));
  }


  // call initialization method for external datagram source
  m_dgsource->init();

  // try to read first event and see if it is a Configure transition
  std::vector<XtcInput::Dgram> eventDg;
  std::vector<XtcInput::Dgram> nonEventDg;
  if (not m_dgsource->next(eventDg, nonEventDg)) {
    // Nothing there at all, this is unexpected
    throw EmptyInput(ERR_LOC);
  }
  
  // for now we only expect one event datagram, can't handle other cases yet
  if (eventDg.size() != 1 or not nonEventDg.empty()) {
    throw UnexpectedInput(ERR_LOC);
  }

  Dgram dg = eventDg.front();
  Dgram::ptr dgptr = dg.dg();

  MsgLog(name(), debug, name() << ": read first datagram, transition = "
        << Pds::TransitionId::name(dgptr->seq.service()));

  if ( dgptr->seq.service() != Pds::TransitionId::Configure ) {
    // Something else than Configure, store if for event()
    MsgLog(name(), warning, "Expected Configure transition for first datagram, received "
           << Pds::TransitionId::name(dgptr->seq.service()) );
    m_putBack = dg;
    return;
  }
  
  m_transitions[dgptr->seq.service()] = dgptr->seq.clock();
  
  // send it to workers
  send(dg);

}

InputModule::Status 
XtcMPMasterInputBase::event(Event& evt, Env& env)
{
  bool done = false;
  while (not done) {

    // get datagram either from saved event or queue
    Dgram dg;
    if (not m_putBack.empty()) {

      dg = m_putBack;
      m_putBack = Dgram();

    } else {

      std::vector<XtcInput::Dgram> eventDg;
      std::vector<XtcInput::Dgram> nonEventDg;
      if (not m_dgsource->next(eventDg, nonEventDg)) {
        // finita
        MsgLog(name(), debug, "EOF seen");
        done = true;
        break;
      }

      // for now we only expect one event datagram, can't handle other cases yet
      if (eventDg.size() != 1 or not nonEventDg.empty()) {
        throw UnexpectedInput(ERR_LOC);
      }

      dg = eventDg.front();
    }
  
    const Pds::Sequence& seq = dg.dg()->seq ;
    const Pds::ClockTime& clock = seq.clock() ;
    Pds::TransitionId::Value trans = seq.service();

    MsgLog(name(), debug, name() << ": found new datagram, transition = "
          << Pds::TransitionId::name(trans));

    switch (trans) {
    
    case Pds::TransitionId::Configure:
      if (not (clock == m_transitions[trans])) {
        MsgLog(name(), warning, name() << ": Multiple Configure transitions encountered");
        m_transitions[trans] = clock;
      } else {
        dg = Dgram();
      }
      break;
      
    case Pds::TransitionId::Unconfigure:
      // do not need this
      dg = Dgram();
      break;
   
    case Pds::TransitionId::BeginRun:
    case Pds::TransitionId::EndRun:
    case Pds::TransitionId::BeginCalibCycle:
    case Pds::TransitionId::EndCalibCycle:
      // do not send repeated datagrams
      if (clock == m_transitions[trans]) {
        dg = Dgram();
      } else {
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::L1Accept:
      // regular event

      if (m_l3tAcceptOnly and not ::l3accept(*dg.dg())) {

        // did not pass L3, its payload is usually empty but if there is Epics
        // data in the event it may be preserved, so try to save it
        memorizeEpics(dg);

      } else if (m_skipEpics and ::epicsOnly(&(dg.dg()->xtc))) {

        // datagram is likely filtered, has only epics data and users do not need to
        // see it. Do not count it as an event too, just save EPICS data and move on.
        memorizeEpics(dg);

      } else if (m_maxEvents and m_l1Count >= m_skipEvents+m_maxEvents) {

        // reached event limit, will go in simulated end-of-run
        MsgLog(name(), debug, name() << ": event limit reached, stopping");
        dg = Dgram();
        done = true;

      } else if (m_l1Count < m_skipEvents) {

        // skipping the events, note that things like environment and EPICS need to be updated
        MsgLog(name(), debug, name() << ": skipping event");
        memorizeEpics(dg);
        dg = Dgram();
        ++m_l1Count;

      } else {

        // take it and count it
        ++m_l1Count;
      }

      break;
    
    case Pds::TransitionId::Unknown:
    case Pds::TransitionId::Reset:
    case Pds::TransitionId::Map:
    case Pds::TransitionId::Unmap:
    case Pds::TransitionId::Enable:
    case Pds::TransitionId::Disable:
    case Pds::TransitionId::NumberOf:
      // Do not do anything for these transitions, just go to next
      dg = Dgram();
      break;
    }

    if (not dg.empty()) {
      // send it to worker(s)
      this->send(dg);
    }

  }


  // we are done on our side, send final empty packet and close all pipes so that workers know it's time to finish
  this->send(Dgram());
  for (std::map<int, psana::MPWorkerId>::const_iterator it = m_workers.begin(); it != m_workers.end(); ++ it) {
    ::close(it->second.fdDataPipe());
  }

  
  return Stop;
}

/// Method which is called once at the end of the job
void 
XtcMPMasterInputBase::endJob(Event& evt, Env& env)
{
}



// send datagram to workers
void
XtcMPMasterInputBase::send(const XtcInput::Dgram& dg)
{
  Dgram::ptr dgptr = dg.dg();

  if (not dgptr) {
    // special case, send final packet to everybody
    for (std::map<int, psana::MPWorkerId>::const_iterator it = m_workers.begin(); it != m_workers.end(); ++ it) {
      sendToWorker(it->first, dg);
    }
    return;
  }

  // if current datagram contain epics data use it instead of memorized ones
  XtcInput::XtcIterator iter(&dgptr->xtc);
  while (Pds::Xtc* xtc = iter.next()) {
    if (xtc->contains.id() == Pds::TypeId::Id_Epics) {
      m_epicsDgs.erase(xtc->src);
    }
  }

  // Keep only EPICS parts from those memorized datagrams
  for (std::map<Pds::Src, XtcInput::Dgram>::iterator it = m_epicsDgs.begin(); it != m_epicsDgs.begin(); ++ it) {
    Dgram::ptr ddg = it->second.dg();
    if (not ::epicsOnly(&(ddg->xtc))) {
      // make a buffer big enough
      size_t bufSize = ddg->xtc.sizeofPayload() + sizeof(Pds::Dgram);
      char* buf = new char[bufSize];

      XtcInput::XtcFilter<XtcInput::XtcFilterTypeIdSrc> filter(XtcFilterTypeIdSrc(Pds::TypeId::Id_Epics, it->first));
      filter.filter(ddg.get(), buf);

      Dgram filtered(Dgram::make_ptr((Pds::Dgram*)buf), it->second.file());
      it->second = filtered;
    }
  }


  if (dgptr->seq.service() == Pds::TransitionId::L1Accept) {

    // L1 Accept is sent to next available worker

    unsigned char workerId = 0;
    ssize_t bytesread = ::read(m_fdReadyPipe, &workerId, 1);
    if (bytesread < 0) {
      throw ExceptionErrno(ERR_LOC, "reading from ready pipe failed");
    } else if (bytesread == 0) {
      // all workers disconnected
      throw GenericException(ERR_LOC, "All worker processes disconnected prematurely");
    }

    sendToWorker(workerId, dg);

    // also may need to remember EPICS stuff from current event
    memorizeEpics(dg);

  } else {

    // other types of datagrams go to every worker

    for (std::map<int, psana::MPWorkerId>::const_iterator it = m_workers.begin(); it != m_workers.end(); ++ it) {
      sendToWorker(it->first, dg);
    }

    // check that there is somebody left
    if (m_workers.empty()) {
      throw GenericException(ERR_LOC, "All worker processes disconnected prematurely");
    }

  }


}

// send datagram to one worker
void
XtcMPMasterInputBase::sendToWorker(int workerId, const XtcInput::Dgram& dg)
{
  Dgram::ptr dgptr = dg.dg();

  MsgLog(name(), debug, "sending " << (dgptr ? Pds::TransitionId::name(dg.dg()->seq.service()) : "final") << " datagram to worker #" << workerId);

  // get data pipe FD for this worker
  std::map<int, psana::MPWorkerId>::iterator witer = m_workers.find(workerId);
  if (witer == m_workers.end()) {
    throw GenericException(ERR_LOC, "Unexpected worker number, not in the worker list");
  }
  int fd = witer->second.fdDataPipe();

  XtcMPDgramSerializer ser(fd);

  // make two lists with datagrams
  std::vector<XtcInput::Dgram> eventDg;
  std::vector<XtcInput::Dgram> nonEventDg;
  if (dgptr) {
    eventDg.push_back(dg);
    for (std::map<Pds::Src, XtcInput::Dgram>::const_iterator it = m_epicsDgs.begin(); it != m_epicsDgs.end(); ++ it) {
      nonEventDg.push_back(it->second);
    }
  }

  // send everything down the line
  try {
    ser.serialize(eventDg, nonEventDg);
  } catch (const std::exception& ex) {
    // something bad happened, exclude this guy from our list but try to continue
    MsgLog(name(), warning, "failed to send data to worker: " << ex.what());
    MsgLog(name(), warning, "will exclude worker from further processing");
    ::close(witer->second.fdDataPipe());
    m_workers.erase(witer);
  }
}

// if datagram has EPICs data then remember it in case it may be needed later
void
XtcMPMasterInputBase::memorizeEpics(const XtcInput::Dgram& dg)
{
  Dgram::ptr dgptr = dg.dg();

  // find sources for any epics data
  std::vector<Pds::Src> sources;
  XtcInput::XtcIterator iter(&dgptr->xtc);
  while (Pds::Xtc* xtc = iter.next()) {
    if (xtc->contains.id() == Pds::TypeId::Id_Epics) {
      if (std::find(sources.begin(), sources.end(), xtc->src) == sources.end()) {
        sources.push_back(xtc->src);
      }

    }
  }

  MsgLog(name(), trace, "found " << sources.size() << " epics sources in an event");

  for (std::vector<Pds::Src>::const_iterator it = sources.begin(); it != sources.end(); ++ it) {
    std::map<Pds::Src, XtcInput::Dgram>::iterator srcit = m_epicsDgs.find(*it);
    if (srcit != m_epicsDgs.end()) {
      // if time is later than already stored update it
      if (dgptr->seq.clock() > srcit->second.dg()->seq.clock()) srcit->second = dg;
    } else {
      m_epicsDgs.insert(std::make_pair(*it, dg));
    }
  }
}

} // namespace PSXtcMPInput
