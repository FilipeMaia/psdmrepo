//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcInputModuleBase...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcInput/XtcInputModuleBase.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <iterator>
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/psddl/l3t.ddl.h"
#include "psddl_psana/epics.ddl.h"
#include "PSTime/Time.h"
#include "PSXtcInput/Exceptions.h"
#include "PSXtcInput/XtcEventId.h"
#include "XtcInput/DgramQueue.h"
#include "XtcInput/DgramReader.h"
#include "XtcInput/XtcFileName.h"
#include "XtcInput/XtcIterator.h"
#include "XtcInput/MergeMode.h"

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
  bool l3accept(Pds::Xtc* xtc)
  {
    XtcInput::XtcIterator iter(xtc);
    while (Pds::Xtc* x = iter.next()) {
      if (x->damage.value() == 0 and x->contains.id() == Pds::TypeId::Id_L3TData) {
        if (x->contains.version() == 1) {
          const Pds::L3T::DataV1* l3t = (Pds::L3T::DataV1*)x->payload();
          return l3t->accept();
        }
      }
    }
    // No L3T info means passed
    return true;
  }

}



//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcInput {

//----------------
// Constructors --
//----------------
XtcInputModuleBase::XtcInputModuleBase (const std::string& name)
  : InputModule(name)
  , m_putBack()
  , m_cvt()
  , m_skipEvents(0)
  , m_maxEvents(0)
  , m_skipEpics(false)
  , m_l3tAcceptOnly(true)
  , m_l1Count(0)
  , m_simulateEOR(0)
{
  std::fill_n(m_transitions, int(Pds::TransitionId::NumberOf), Pds::ClockTime(0, 0));

  // get number of events to process/skip from psana configuration
  ConfigSvc::ConfigSvc cfg = configSvc();
  m_skipEvents = cfg.get("psana", "skip-events", 0UL);
  m_maxEvents = cfg.get("psana", "events", 0UL);
  m_skipEpics = cfg.get("psana", "skip-epics", true);
  m_l3tAcceptOnly = cfg.get("psana", "l3t-accept-only", true);
}

//--------------
// Destructor --
//--------------
XtcInputModuleBase::~XtcInputModuleBase ()
{
}

/// Method which is called once at the beginning of the job
void 
XtcInputModuleBase::beginJob(Event& evt, Env& env)
{
  MsgLog(name(), debug, name() << ": in beginJob()");

  // call initialization method for external datagram source
  this->initDgramSource();

  // try to read first event and see if it is a Configure transition
  XtcInput::Dgram dg(this->nextDgram());
  if (dg.empty()) {
    // Nothing there at all, this is unexpected
    throw EmptyInput(ERR_LOC);
  }
  
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
  
  // Store configuration info in the environment
  fillEnv(dg, env);
  fillEventDg(dg, evt);
  fillEventId(dg, evt);
  // there is BLD data in Configure which is event-like data
  fillEvent(dg, evt, env);

}

InputModule::Status 
XtcInputModuleBase::event(Event& evt, Env& env)
{
  MsgLog(name(), debug, name() << ": in event() - m_l1Count=" << m_l1Count
      << " m_maxEvents=" << m_maxEvents << " m_skipEvents=" << m_skipEvents);

  // are we in the simulated EOR/EOF
  if (m_simulateEOR > 0) {
    // fake EndRun, prepare to stop on next call
    MsgLog(name(), debug, name() << ": simulated EOR");
    fillEventId(m_putBack, evt);
    fillEventDg(m_putBack, evt);
    // negative means stop at next call
    m_simulateEOR = -1;
    return EndRun;
  } else if (m_simulateEOR < 0) {
    // fake EOF
    MsgLog(name(), debug, name() << ": simulated EOF");
    return Stop;
  }

  Status status = Skip;
  bool found = false;
  while (not found) {

    // get datagram either from saved event or queue
    XtcInput::Dgram dg;
    if (not m_putBack.empty()) {
      dg = m_putBack;
      m_putBack = Dgram();
    } else {
      dg = this->nextDgram();
    }
  
    if (dg.empty()) {
      // finita
      MsgLog(name(), debug, "EOF seen");
      return Stop;
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
        fillEnv(dg, env);
      }
      break;
      
    case Pds::TransitionId::Unconfigure:
      break;
   
    case Pds::TransitionId::BeginRun:
      // signal new run, content is not relevant
      if (not (clock == m_transitions[trans])) {
        fillEventId(dg, evt);
        fillEventDg(dg, evt);
        status = BeginRun;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::EndRun:
      // signal end of run, content is not relevant
      if (not (clock == m_transitions[trans])) {
        fillEventId(dg, evt);
        fillEventDg(dg, evt);
        status = EndRun;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::BeginCalibCycle:
      // copy config data and signal new calib cycle
      if (not (clock == m_transitions[trans])) {
        fillEnv(dg, env);
        fillEventId(dg, evt);
        fillEventDg(dg, evt);
        status = BeginCalibCycle;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::EndCalibCycle:
      // stop calib cycle
      if (not (clock == m_transitions[trans])) {
        fillEventId(dg, evt);
        fillEventDg(dg, evt);
        status = EndCalibCycle;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::L1Accept:
      // regular event

      if (m_l3tAcceptOnly and not ::l3accept(&(dg.dg()->xtc))) {

        // did not pass L3, its payload is usually empty but if there is Epics
        // data in the event it may be preserved, so try to save it
        fillEnv(dg, env);

      } else if (m_skipEpics and ::epicsOnly(&(dg.dg()->xtc))) {

        // datagram is likely filtered, has only epics data and users do not need to
        // see it. Do not count it as an event too, just save EPICS data and move on.
        fillEnv(dg, env);

      } else if (m_maxEvents and m_l1Count >= m_skipEvents+m_maxEvents) {

        // reached event limit, will go in simulated end-of-run
        MsgLog(name(), debug, name() << ": event limit reached, simulated EndCalibCycle");
        fillEventId(dg, evt);
        fillEventDg(dg, evt);
        found = true;
        status = EndCalibCycle;
        m_simulateEOR = 1;
        // remember datagram to be used in simulated EndRun
        m_putBack = dg;

      } else if (m_l1Count < m_skipEvents) {

        // skipping the events, note that things like environment and EPICS need to be updated
        MsgLog(name(), debug, name() << ": skipping event");
        fillEnv(dg, env);
        found = true;
        status = Skip;

        ++m_l1Count;

      } else {

        fillEnv(dg, env);
        fillEvent(dg, evt, env);
        fillEventId(dg, evt);
        fillEventDg(dg, evt);
        found = true;
        status = DoEvent;

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
      break;
    }
  }
  
  return status ;
}

/// Method which is called once at the end of the job
void 
XtcInputModuleBase::endJob(Event& evt, Env& env)
{
}

// Fill event with datagram contents
void 
XtcInputModuleBase::fillEvent(const XtcInput::Dgram& dg, Event& evt, Env& env)
{
  MsgLog(name(), debug, name() << ": in fillEvent()");

  Dgram::ptr dgptr = dg.dg();
  
  // Loop over all XTC contained in the datagram
  XtcInput::XtcIterator iter(&dgptr->xtc);
  while (Pds::Xtc* xtc = iter.next()) {
      
    boost::shared_ptr<Pds::Xtc> xptr(dgptr, xtc);
    // call the converter which will fill event with data
    m_cvt.convert(xptr, evt, env.configStore());
    
  }
}

// Fill event with datagram contents
void
XtcInputModuleBase::fillEventId(const XtcInput::Dgram& dg, Event& evt)
{
  MsgLog(name(), debug, name() << ": in fillEventId()");

  Dgram::ptr dgptr = dg.dg();

  const Pds::Sequence& seq = dgptr->seq ;
  const Pds::ClockTime& clock = seq.clock() ;

  // Store event ID
  PSTime::Time evtTime(clock.seconds(), clock.nanoseconds());
  unsigned run = dg.file().run();
  unsigned fiducials = seq.stamp().fiducials();
  unsigned ticks = seq.stamp().ticks();
  unsigned vect = seq.stamp().vector();
  boost::shared_ptr<PSEvt::EventId> eventId = boost::make_shared<XtcEventId>(run, evtTime, fiducials, ticks, vect);
  evt.put(eventId);
}

// Fill event with datagram contents
void
XtcInputModuleBase::fillEventDg(const XtcInput::Dgram& dg, Event& evt)
{
  MsgLog(name(), debug, name() << ": in fillEventDg()");

  Dgram::ptr dgptr = dg.dg();

  // Store datagram itself in the event
  evt.put(dgptr);
}

// Fill environment with datagram contents
void 
XtcInputModuleBase::fillEnv(const XtcInput::Dgram& dg, Env& env)
{
  MsgLog(name(), debug, name() << ": in fillEnv()");

  // All objects in datagram in Configuration and BeginCalibCycle transitions
  // (except for EPICS data) are considered configuration data. Just store them
  // them in the ConfigStore part of the environment

  Dgram::ptr dgptr = dg.dg();

  const Pds::Sequence& seq = dgptr->seq ;
  if (seq.service() == Pds::TransitionId::Configure or 
      seq.service() == Pds::TransitionId::BeginCalibCycle) {
    
    // Loop over all XTC contained in the datagram
    XtcInput::XtcIterator iter(&dgptr->xtc);
    while (Pds::Xtc* xtc = iter.next()) {

      if (xtc->contains.id() != Pds::TypeId::Id_Epics) {
        boost::shared_ptr<Pds::Xtc> xptr(dgptr, xtc);
        // call the converter which will fill config store
        m_cvt.convertConfig(xptr, env.configStore());
      }

      if (xtc->contains.id() == Pds::TypeId::Id_EpicsConfig) {
        // need to tell Epics store about aliases
        boost::shared_ptr<Psana::Epics::ConfigV1> cfgV1 = env.configStore().get(xtc->src);
        if (cfgV1) {
          const ndarray<const Psana::Epics::PvConfigV1, 1>& pvs = cfgV1->getPvConfig();
          for (unsigned i = 0; i != pvs.shape()[0]; ++ i) {
            const Psana::Epics::PvConfigV1& pvcfg = pvs[i];
            env.epicsStore().storeAlias(xtc->src, pvcfg.pvId(), pvcfg.description());
          }
        }
      }

    }
      
  }

  // Convert EPICS too and store it in EPICS store
  // Loop over all XTC contained in the datagram
  XtcInput::XtcIterator iter(&dgptr->xtc);
  while (Pds::Xtc* xtc = iter.next()) {

    if (xtc->contains.id() == Pds::TypeId::Id_Epics) {
      // call the converter which will fill config store
      boost::shared_ptr<Pds::Xtc> xptr(dgptr, xtc);
      m_cvt.convertEpics(xptr, env.epicsStore());
    }
    
  }

}

} // namespace PSXtcInput
