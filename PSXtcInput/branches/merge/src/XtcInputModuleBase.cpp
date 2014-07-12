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
#include <climits>
#include <algorithm>
#include <iterator>
#include <vector>
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "pdsdata/xtc/L1AcceptEnv.hh"
#include "pdsdata/psddl/alias.ddl.h"
#include "psddl_psana/epics.ddl.h"
#include "PSTime/Time.h"
#include "PSXtcInput/Exceptions.h"
#include "PSXtcInput/XtcEventId.h"
#include "XtcInput/DgramQueue.h"
#include "XtcInput/DgramReader.h"
#include "XtcInput/XtcFileName.h"
#include "XtcInput/XtcIterator.h"
#include "XtcInput/MergeMode.h"
#include "XtcInput/DgramList.h"

#include "PSEvt/DamageMap.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace XtcInput;

namespace {


  const char * logger = "XtcInputModuleBase";

  // return true if all datagrams contains EPICS data only
  bool epicsOnly(const std::vector<XtcInput::Dgram>& dgs)
  {
    BOOST_FOREACH(const XtcInput::Dgram& dg, dgs) {
      XtcInput::XtcIterator iter(&(dg.dg()->xtc));
      while (Pds::Xtc* x = iter.next()) {
        switch (x->contains.id()) {
        case Pds::TypeId::Id_Xtc:
        case Pds::TypeId::Id_Epics:
          continue;
        default:
          return false;
        }
      }
    }
    return true;
  }

  // return true if event passed L3 selection (of there was no L3 defined)
  bool l3accept(const std::vector<XtcInput::Dgram>& dgs)
  {
    // if at least one is not trimmed then we accept all
    BOOST_FOREACH(const XtcInput::Dgram& dg, dgs) {
      if (not static_cast<const Pds::L1AcceptEnv&>(dg.dg()->env).trimmed()) return true;
    }
    return false;
  }

  long nextNonNegativeValue(const long v) {
    if ((v >=0) and (v < LONG_MAX)) return (v+1);
    return 0;
  }

  class LessStream {
  public:
    LessStream() {}
    bool operator()(const XtcInput::Dgram &a, const XtcInput::Dgram &b) {
      return a.file().stream() < b.file().stream();
    }
  };

  bool allDgsHaveSameTransition(const std::vector<XtcInput::Dgram> &dgs) {
    Pds::TransitionId::Value last = Pds::TransitionId::Unknown;
    BOOST_FOREACH(const XtcInput::Dgram& dg, dgs) {
      if (not dg.empty()) {
        if ((last != Pds::TransitionId::Unknown) and (last != dg.dg()->seq.service())) {
          return false;
        }
        last = dg.dg()->seq.service();
      }
    }
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
XtcInputModuleBase::XtcInputModuleBase (const std::string& name,
    const boost::shared_ptr<IDatagramSource>& dgsource, bool noSkip)
  : InputModule(name)
  , m_dgsource(dgsource)
  , m_damagePolicy()
  , m_putBack()
  , m_cvt()
  , m_skipEvents(0)
  , m_maxEvents(0)
  , m_skipEpics(true)
  , m_l3tAcceptOnly(true)
  , m_l1Count(0)
  , m_eventTagEpicsStore(0)
  , m_simulateEOR(0)
  , m_run(-1)
{
  std::fill_n(m_transitions, int(Pds::TransitionId::NumberOf), Pds::ClockTime(0, 0));

  if (not noSkip) {
    // get number of events to process/skip from psana configuration
    ConfigSvc::ConfigSvc cfg = configSvc();
    m_skipEvents = cfg.get("psana", "skip-events", 0UL);
    m_maxEvents = cfg.get("psana", "events", 0UL);
    m_skipEpics = cfg.get("psana", "skip-epics", true);
    m_l3tAcceptOnly = cfg.get("psana", "l3t-accept-only", true);
  }
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

  m_eventTagEpicsStore = nextNonNegativeValue(m_eventTagEpicsStore);

  // call initialization method for external datagram source
  m_dgsource->init();

  // Read initial datagrams, skip all Map transitions, stop at first non-Map.
  // If first non-Map is Configure then update event/env, otherwise push
  // it into read_back buffer.
  bool foundNonMap = false;
  for (int count = 0; not foundNonMap; ++ count) {

    std::vector<XtcInput::Dgram> eventDg;
    std::vector<XtcInput::Dgram> nonEventDg;
    if (not m_dgsource->next(eventDg, nonEventDg)) {
      if (count == 0) {
        // Nothing there at all, this can happen with indexing, since
        // the user has not called "jump" yet.
        return;
      } else {
        // just stop here
        break;
      }
    }

    // If the same EventKey is in both the DAQ and Control stream,
    // we want the DAQ stream to take precedence.
    // For the Env, we can do this by processing the DAQ stream last 
    // (it will overwrite the entry from the Control stream).
    // For the Event, we process the DAQ stream first, then an error will
    // be generated when processing the Control stream.
    sort(eventDg.begin(), eventDg.end(), LessStream());
    sort(nonEventDg.begin(), nonEventDg.end(), LessStream());

    // push non-event stuff to environment. 
    // Iterate in reverse to process DAQ streams last and overwrite 
    // duplicate entries with the Control streams.
    // Usually first transition (Configure) should not have any non-event attached data
    BOOST_REVERSE_FOREACH(const XtcInput::Dgram& dg, nonEventDg) {
      fillEnv(dg, env);
    }

    MsgLog(logger,trace,"beginJob datagrams: ");
    int idx=0;
    BOOST_FOREACH(const XtcInput::Dgram& dg, eventDg) {
      MsgLog(logger,debug,"  dg " << idx << ": " << Dgram::dumpStr(dg));
      ++idx;
    }

    if (not allDgsHaveSameTransition(eventDg)) {
      MsgLog(name(), warning, name() << "first datagrams do not have same transition.");
    }

    Pds::TransitionId::Value transition = Pds::TransitionId::Map;
    if (not eventDg.empty()) {
      transition = eventDg.front().dg()->seq.service();
    }

    MsgLog(name(), debug, name() << ": read first datagram(s), transition = "
          << Pds::TransitionId::name(transition));

    // skip Map transition
    foundNonMap = transition != Pds::TransitionId::Map;
    if (not foundNonMap) continue;

    // If this is not Map then we expect Configure here, anything else must be handled in event()
    if (transition != Pds::TransitionId::Configure) {
      // Something else than Configure, store if for event()
      MsgLog(name(), warning, "Expected Configure transition for first datagram, received "
             << Pds::TransitionId::name(transition) );
      m_putBack = eventDg;
      break;
    }

    // get the transition clock time, event id, and event datagram from the first
    // event in the list which should have the smallest stream number based on prior sort. 
    XtcInput::Dgram firstDg = eventDg.front();

    m_transitions[firstDg.dg()->seq.service()] = firstDg.dg()->seq.clock();

    fillEventDgList(eventDg, evt);
    fillEventId(firstDg, evt);

    boost::shared_ptr<PSEvt::DamageMap> damageMap = boost::make_shared<PSEvt::DamageMap>();
    evt.put(damageMap);

    // process in reverse to get DAQ streams after control streams
    BOOST_REVERSE_FOREACH(const XtcInput::Dgram& dg, eventDg) {
      // Store configuration info in the environment
      fillEnv(dg, env);
    }

    // process in in order to get DAQ streams before control streams
    BOOST_FOREACH(const XtcInput::Dgram& dg, eventDg) {
      // there is BLD data in Configure which is event-like data
      fillEvent(dg, evt, env);
    }
  }
}

InputModule::Status 
XtcInputModuleBase::event(Event& evt, Env& env)
{
  MsgLog(name(), debug, name() << ": in event() - m_l1Count=" << m_l1Count
      << " m_maxEvents=" << m_maxEvents << " m_skipEvents=" << m_skipEvents);

  m_eventTagEpicsStore = nextNonNegativeValue(m_eventTagEpicsStore);
  // are we in the simulated EOR/EOF
  if (m_simulateEOR > 0) {
    // fake EndRun, prepare to stop on next call
    MsgLog(name(), debug, name() << ": simulated EOR");
    if (m_putBack.size()) {
      fillEventId(m_putBack[0], evt);
      fillEventDgList(m_putBack, evt);
    }
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

    std::vector<XtcInput::Dgram> eventDg;
    std::vector<XtcInput::Dgram> nonEventDg;

    // get datagram either from saved event or queue
    if (not m_putBack.empty()) {

      std::swap(eventDg, m_putBack);
      m_putBack.clear();

    } else {

      if (not m_dgsource->next(eventDg, nonEventDg)) {
        // finita
        MsgLog(name(), debug, "EOF seen");
        return Stop;
      }
      // sort by stream number to get DAQ stream in the front
      sort(eventDg.begin(), eventDg.end(), LessStream());
      sort(nonEventDg.begin(), nonEventDg.end(), LessStream());

    }

    // push all non-event stuff into environment, reverse order to 
    // process DAQ streams after Control streams
    BOOST_REVERSE_FOREACH(const XtcInput::Dgram& dg, nonEventDg) {
      fillEnv(dg, env);
    }

    if (eventDg.empty()) {
      // can't do anything, skip to next transition
      continue;
    }

    if (not allDgsHaveSameTransition(eventDg)) {
      MsgLog(name(), warning, name() 
             << ": eventDg's do not all have the same transition. Using first transition.");
      // print the dgram headers since they don't have the same transition
      int idx = 0;
      BOOST_FOREACH(const XtcInput::Dgram& dg, eventDg) {
        MsgLog(name(),info,"  dg " << idx <<": " << Dgram::dumpStr(dg));
        ++idx;
      }
    }
    // We use the first datagram in eventDg for the transition, and clock.
    // The clock will differ between DAQ and control streams - 
    const Pds::Sequence& seq = eventDg.front().dg()->seq;
    const Pds::ClockTime& clock = seq.clock();
    Pds::TransitionId::Value trans = seq.service();
    
    MsgLog(name(), debug, name() << ": found " << eventDg.size() << " new datagram(s), transition = "
           << Pds::TransitionId::name(trans));
    
    switch (trans) {
    
    case Pds::TransitionId::Configure:
      if (not (clock == m_transitions[trans])) {
        m_transitions[trans] = clock;
        BOOST_REVERSE_FOREACH(const XtcInput::Dgram& dg, eventDg) {
          fillEnv(dg, env);
        }
      }
      break;
      
    case Pds::TransitionId::Unconfigure:
      break;
   
    case Pds::TransitionId::BeginRun:
      // take run number from transition env, in some streams env is not set (or set to 0), so we want to skip those
      BOOST_FOREACH(const XtcInput::Dgram& dg, eventDg) {
        if (dg.dg()->env.value() > 0) {
          m_run = dg.dg()->env.value();
          break;
        }
      }
      // signal new run, content is not relevant
      if (not (clock == m_transitions[trans])) {
        fillEventId(eventDg.front(), evt);
        fillEventDgList(eventDg, evt);
        status = BeginRun;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::EndRun:
      // signal end of run, content is not relevant
      if (not (clock == m_transitions[trans])) {
        fillEventId(eventDg.front(), evt);
        fillEventDgList(eventDg, evt);
        // reset run number, so that if next BeginRun is missing we don't reuse this run
        m_run = -1;
        status = EndRun;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::BeginCalibCycle:
      // copy config data and signal new calib cycle
      if (not (clock == m_transitions[trans])) {
        BOOST_REVERSE_FOREACH(const XtcInput::Dgram& dg, eventDg) {
          fillEnv(dg, env);
        }
        fillEventId(eventDg.front(), evt);
        fillEventDgList(eventDg, evt);
        status = BeginCalibCycle;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::EndCalibCycle:
      // stop calib cycle
      if (not (clock == m_transitions[trans])) {
        fillEventId(eventDg.front(), evt);
        fillEventDgList(eventDg, evt);
        status = EndCalibCycle;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::L1Accept:
      // regular event

      if (m_l3tAcceptOnly and not ::l3accept(eventDg)) {

        // did not pass L3, its payload is usually empty but if there is Epics
        // data in the event it may be preserved, so try to save it
        BOOST_REVERSE_FOREACH(const XtcInput::Dgram& dg, eventDg) {
          fillEnv(dg, env);
        }

      } else if (m_skipEpics and ::epicsOnly(eventDg)) {

        // datagram is likely filtered, has only epics data and users do not need to
        // see it. Do not count it as an event too, just save EPICS data and move on.
        BOOST_REVERSE_FOREACH(const XtcInput::Dgram& dg, eventDg) {
          fillEnv(dg, env);
        }

      } else if (m_maxEvents and m_l1Count >= m_skipEvents+m_maxEvents) {

        // reached event limit, will go in simulated end-of-run
        MsgLog(name(), debug, name() << ": event limit reached, simulated EndCalibCycle");
        fillEventId(eventDg.front(), evt);
        fillEventDgList(eventDg, evt);
        found = true;
        status = EndCalibCycle;
        m_simulateEOR = 1;
        // remember datagram to be used in simulated EndRun
        m_putBack = eventDg;

      } else if (m_l1Count < m_skipEvents) {

        // skipping the events, note that things like environment and EPICS need to be updated
        MsgLog(name(), debug, name() << ": skipping event");
        BOOST_REVERSE_FOREACH(const XtcInput::Dgram& dg, eventDg) {
          fillEnv(dg, env);
        }
        found = true;
        status = Skip;

        ++m_l1Count;

      } else {

        boost::shared_ptr<PSEvt::DamageMap> damageMap = boost::make_shared<PSEvt::DamageMap>();
        evt.put(damageMap);
        BOOST_REVERSE_FOREACH(const XtcInput::Dgram& dg, eventDg) {
          fillEnv(dg, env);
        }
        BOOST_FOREACH(const XtcInput::Dgram& dg, eventDg) {
          fillEvent(dg, evt, env);
        }
        fillEventId(eventDg.front(), evt);
        fillEventDgList(eventDg, evt);
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
  
  boost::shared_ptr<PSEvt::DamageMap> damageMap = evt.get();
  // Loop over all XTC contained in the datagram
  XtcInput::XtcIterator iter(&dgptr->xtc);
  while (Pds::Xtc* xtc = iter.next()) {
    const Pds::TypeId& typeId = xtc->contains;
    const Pds::Damage damage = xtc->damage;
    if (typeId.id() == Pds::TypeId::Any) {
      if (damage.value()) {
        damageMap->addSrcDamage(xtc->src,damage);
      } else {
        MsgLog(name(), warning, name() << ": unexpected - xtc type id is 'Any' but its damage=0");
      }
      continue;
    }
    std::vector<const std::type_info *> convertTypeInfoPtrs = m_cvt.getConvertTypeInfoPtrs(typeId);
    for (unsigned idx=0; idx < convertTypeInfoPtrs.size(); ++idx) {
      (*damageMap)[PSEvt::EventKey( convertTypeInfoPtrs[idx], xtc->src, "") ] = damage;
    }
    bool storeObject = m_damagePolicy.eventDamagePolicy(damage, typeId.id());
    if (not storeObject) {
      MsgLog(name(),debug,name() << "damage = " << damage.value() << " typeId=" 
             << typeId.id() << " src=" << xtc->src << " not storing in Event");
      continue;
    }

    boost::shared_ptr<Pds::Xtc> xptr(dgptr, xtc);   
    m_cvt.convert(xptr, evt, env.configStore());
    
  }
}

void
XtcInputModuleBase::fillEventId(const XtcInput::Dgram& dg, Event& evt)
{
  MsgLog(name(), debug, name() << ": in fillEventId()");

  Dgram::ptr dgptr = dg.dg();

  const Pds::Sequence& seq = dgptr->seq ;
  const Pds::ClockTime& clock = seq.clock() ;

  // Store event ID
  PSTime::Time evtTime(clock.seconds(), clock.nanoseconds());
  unsigned run = m_run > 0 ? int(m_run) : dg.file().run();
  unsigned fiducials = seq.stamp().fiducials();
  unsigned ticks = seq.stamp().ticks();
  unsigned vect = seq.stamp().vector();
  unsigned control = seq.stamp().control();
  boost::shared_ptr<PSEvt::EventId> eventId = boost::make_shared<XtcEventId>(run, evtTime, fiducials, ticks, vect, control);
  evt.put(eventId);
}

void
XtcInputModuleBase::fillEventDgList(const std::vector<XtcInput::Dgram> & dgList, Event& evt)
{
  MsgLog(name(), debug, name() << ": in fillEventDgList()");

  boost::shared_ptr< XtcInput::DgramList > dgramList = 
    boost::make_shared< XtcInput::DgramList >();

  BOOST_FOREACH(const XtcInput::Dgram & dg, dgList) {
    dgramList->push_back(dg);
  }

  // Store list of datagrams in the event
  evt.put(dgramList);
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

    // before we start adding all config types we need to update alias map so
    // that when we add objects to proxy dict correct version of alias map is used
    if (env.aliasMap()) {
      XtcInput::XtcIterator iter1(&dgptr->xtc);
      while (Pds::Xtc* xtc = iter1.next()) {
        if (xtc->contains.id() == Pds::TypeId::Id_AliasConfig) {

          boost::shared_ptr<PSEvt::AliasMap> amap = env.aliasMap();

          if (xtc->contains.version() == 1) {
            const Pds::Alias::ConfigV1* cfgV1 = (const Pds::Alias::ConfigV1*)xtc->payload();
            const ndarray<const Pds::Alias::SrcAlias, 1>& aliases = cfgV1->srcAlias();
            for (unsigned i = 0; i != aliases.shape()[0]; ++ i) {
              const Pds::Alias::SrcAlias& alias = aliases[i];
              amap->add(alias.aliasName(), alias.src());
            }
          } else {
            MsgLog(name(), warning, name() << "failed to find Alias::ConfigV1 in config store");
          }
        }
      }
    }
    
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
      m_cvt.convertEpics(xptr, env.epicsStore(), m_eventTagEpicsStore);
    }
    
  }

}

} // namespace PSXtcInput
