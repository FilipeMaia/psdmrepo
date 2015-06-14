//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class XtcInputModuleBase...
//
// Author List:
//     Andrei Salnikov
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
#include <string>

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
#include "XtcInput/XtcFileName.h"
#include "XtcInput/XtcIterator.h"
#include "XtcInput/MergeMode.h"
#include "XtcInput/DgramList.h"
#include "XtcInput/DgramUtil.h"
#include "PSEvt/DamageMap.h"
#include "IData/Dataset.h"
#include "psddl_pds2psana/SmallDataProxy.h"
#include "PSEvt/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace XtcInput;

namespace {


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


  long nextNonNegativeValue(const long v) {
    if ((v >=0) and (v < LONG_MAX)) return (v+1);
    return 0;
  }

  bool isConfigOrBeginCalib(const XtcInput::Dgram &dg) {
    if (dg.empty()) return false;
    XtcInput::Dgram::ptr dgptr = dg.dg();
    if (not dgptr) return false;
    const Pds::Sequence& seq = dgptr->seq ;
    if ((seq.service() == Pds::TransitionId::Configure) or
        (seq.service() == Pds::TransitionId::BeginCalibCycle)) {
      return true;
    }
    return false;
  }

  // returns true if typeId is for Xtc Any
  inline bool checkForAndRecordSrcDamage(const Pds::TypeId& typeId, Pds::Xtc *xtc, 
                                         const Pds::Damage &damage, boost::shared_ptr<PSEvt::DamageMap> &damageMap, 
                                         const char *loggerName) {
    if (typeId.id() == Pds::TypeId::Any) {
      if (damage.value()) {
        damageMap->addSrcDamage(xtc->src,damage);
      } else {
        MsgLog(loggerName, warning, loggerName << ": unexpected - xtc type id is 'Any' but its damage=0");
      }
      return true;
    }
    return false;
  }

}



//             ----------------------------------------
//             -- Public Function Member Definitions --
//             ----------------------------------------

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
  , m_firstControlStream(80)
  , m_l1Count(0)
  , m_eventTagEpicsStore(0)
  , m_simulateEOR(0)
  , m_run(-1)
  , m_liveMode(false)
{
  std::fill_n(m_transitions, int(Pds::TransitionId::NumberOf), Pds::ClockTime(0, 0));

  ConfigSvc::ConfigSvc cfg = configSvc();
  m_firstControlStream = cfg.get("psana", "first_control_stream", m_firstControlStream);
  if (not noSkip) {
    // get number of events to process/skip from psana configuration
    m_skipEvents = cfg.get("psana", "skip-events", 0UL);
    m_maxEvents = cfg.get("psana", "events", 0UL);
    m_skipEpics = cfg.get("psana", "skip-epics", true);
    m_l3tAcceptOnly = cfg.get("psana", "l3t-accept-only", true);
  }
  try {
    std::list<std::string> files = configList("files");
    
    // The DgramReader does a more thorough job of checking the files input, throwing errors
    // if live is mixed with dead mode, etc, here if we find "live" in one of the files, we assume
    // live mode 

    // TODO: Presently - LiveFilesDB does not know how to get small data filenames - so if live mode is 
    // present, and smd is as well, we'd like to warn the user
    bool smallData = false;
    for (std::list<std::string>::iterator file = files.begin(); file != files.end(); ++file) {
      IData::Dataset ds(*file);
      if (ds.exists("live")) {
        m_liveMode = true;
      }
      if (ds.exists("smd")) {
        smallData = true;
      }
    }
  } catch (ConfigSvc::Exception &) {
    MsgLog(name, error, " " << name << ": unable to read 'files' parameters, assuming non-live mode");
  }
  m_liveTimeOut = config("liveTimeout", 120U);
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
        // Nothing there at all, this is unexpected
        throw EmptyInput(ERR_LOC);
      } else {
        // just stop here
        break;
      }
    }

    // fillEnv and fillEvent require the datagrams to have DAQ streams
    // first, and Control streams second. We want the EventId to come from
    // a DAQ stream if one is present. After sorting by stream number, we
    // can use the first datagram for the EventId.
    sort(eventDg.begin(), eventDg.end(), LessStream());
    sort(nonEventDg.begin(), nonEventDg.end(), LessStream());

    // push non-event stuff to environment. 
    fillEnv(nonEventDg, env);

    MsgLog(name(),trace," beginJob datagrams: ");
    int idx=0;
    BOOST_FOREACH(const XtcInput::Dgram& dg, eventDg) {
      MsgLog(name(),debug,"  dg " << idx << ": " << Dgram::dumpStr(dg));
      ++idx;
    }

    if (not allDgsHaveSameTransition(eventDg)) {
      MsgLog(name(), warning, name() << ": first datagrams do not have same transition.");
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
      MsgLog(name(), warning, ": Expected Configure transition for first datagram, received "
             << Pds::TransitionId::name(transition) );
      m_putBack = eventDg;
      break;
    }

    // get the transition clock time, event id, and event datagram from the first
    // event in the list. Given the prior sort by stream number, this will be a DAQ datagram 
    // if any DAQ streams are present in this event
    XtcInput::Dgram firstDg = eventDg.front();

    m_transitions[firstDg.dg()->seq.service()] = firstDg.dg()->seq.clock();

    fillEventDgList(eventDg, evt);
    fillEventId(firstDg, evt);

    boost::shared_ptr<PSEvt::DamageMap> damageMap = boost::make_shared<PSEvt::DamageMap>();
    evt.put(damageMap);

    fillEnv(eventDg, env);

    // there is BLD data in Configure which is event-like data
    fillEvent(eventDg, evt, env);
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
        MsgLog(name(), debug, ": EOF seen");
        return Stop;
      }
      // sort by stream number to get DAQ streams in front.
      // This is for fillEnv and fillEvent, and obtaining the EventId from a 
      // DAQ datagram (if any are present for this event).
      sort(eventDg.begin(), eventDg.end(), LessStream());
      sort(nonEventDg.begin(), nonEventDg.end(), LessStream());

    }

    fillEnv(nonEventDg, env);

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
    // The clock will differ between DAQ and control streams. Given prior sort, this will give
    // us a DAQ stream if one is present in this event.
    const Pds::Sequence& seq = eventDg.front().dg()->seq;
    const Pds::ClockTime& clock = seq.clock();
    Pds::TransitionId::Value trans = seq.service();
    
    MsgLog(name(), debug, name() << ": found " << eventDg.size() << " new datagram(s), transition = "
           << Pds::TransitionId::name(trans));
    
    switch (trans) {
    
    case Pds::TransitionId::Configure:
      if (not (clock == m_transitions[trans])) {
        m_transitions[trans] = clock;
        fillEnv(eventDg, env);
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
        fillEnv(eventDg, env);
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

      if (m_l3tAcceptOnly and not l3tAcceptPass(eventDg, m_firstControlStream)) {

        // did not pass L3, its payload is usually empty but if there is Epics
        // data in the event it may be preserved, so try to save it
        fillEnv(eventDg, env);

      } else if (m_skipEpics and ::epicsOnly(eventDg)) {

        // datagram is likely filtered, has only epics data and users do not need to
        // see it. Do not count it as an event too, just save EPICS data and move on.
        fillEnv(eventDg, env);

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
        fillEnv(eventDg, env);
        found = true;
        status = Skip;

        ++m_l1Count;

      } else {

        boost::shared_ptr<PSEvt::DamageMap> damageMap = boost::make_shared<PSEvt::DamageMap>();
        evt.put(damageMap);
        fillEnv(eventDg, env);
        fillEvent(eventDg, evt, env);
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

/// Fill event from list of datagrams 
/// Datagrams should be sorted with DAQ streams first, Control streams last.
/// If transition is Configure or BeginCalibCycle, uses first DAQ stream 
/// (skips others) and all control streams.
void 
XtcInputModuleBase::fillEvent(const std::vector<XtcInput::Dgram>& dgList, Event& evt, Env& env)
{
  // If the same EventKey is in both the DAQ and Control stream,
  // we want the DAQ stream to take precedence so that the DAQ data gets stored.
  // Below we go through all datagrams in order, assuming they have been sorted by 
  // stream number, we will get the DAQ streams first.

  // The Duplicate key exception is currently caught by the dispatch code.

  // if this is a Configure or BeginCalibCycle, then all DAQ dgrams are 
  // duplicates of one another. We need only store one of the DAQ dgrams. 
  // It seems cleaner and less error prone to just store one DAQ dgram in this case.
  int numDaqStored = 0;
  int numControlStored = 0;
  bool storedDaq = false;
  BOOST_FOREACH(const XtcInput::Dgram& dg, dgList) {
    bool isDaq = int(dg.file().stream()) < m_firstControlStream;
    if (isDaq and storedDaq and ::isConfigOrBeginCalib(dg)) continue;
    fillEvent(dg, evt, env);
    if (isDaq) {
      ++numDaqStored;
      storedDaq = true;
    } else {
      ++numControlStored;
    }
  }

  MsgLog(name(), debug, name() << ": in fillEvent() from dgList of " 
         << dgList.size() << " dgrams. Stored "
         << numControlStored << " control dgrams and "
         << numDaqStored << " DAQ dgrams");
}

// Fill event with datagram contents
void 
XtcInputModuleBase::fillEvent(const XtcInput::Dgram& dg, Event& evt, Env& env)
{
  MsgLog(name(), debug, name() << ": in fillEvent()");

  boost::shared_ptr<psddl_pds2psana::SmallDataProxy> smallDataProxy = \
    psddl_pds2psana::SmallDataProxy::makeSmallDataProxy(dg.file(), m_liveMode, m_liveTimeOut, m_cvt,  &evt, env);

  Dgram::ptr dgptr = dg.dg();

  boost::shared_ptr<PSEvt::DamageMap> damageMap = evt.get();
  // Loop over all XTC contained in the datagram
  XtcInput::XtcIterator iter(&dgptr->xtc);
  while (Pds::Xtc* xtc = iter.next()) {
    const Pds::TypeId& typeId = xtc->contains;
    const Pds::Damage damage = xtc->damage;
    bool isXtcAny = checkForAndRecordSrcDamage(typeId, xtc, damage, damageMap, name().c_str());
    if (isXtcAny) continue;
    bool isSmallDataProxy = psddl_pds2psana::SmallDataProxy::isSmallDataProxy(typeId);
    std::vector<const std::type_info *> convertTypeInfoPtrs;
    bool storeObject;

    if (isSmallDataProxy) {
      convertTypeInfoPtrs = psddl_pds2psana::SmallDataProxy::getSmallConvertTypeInfoPtrs(xtc, m_cvt);
      const Pds::TypeId proxiedTypeId = psddl_pds2psana::SmallDataProxy::getSmallDataProxiedType(xtc);
      if (proxiedTypeId.id() == Pds::TypeId::Id_Epics) {
        MsgLog(name(), error, " epics has been proxied in small data - skipping");
        continue;
      }
      storeObject = m_damagePolicy.eventDamagePolicy(damage, proxiedTypeId.id());
    } else {
      convertTypeInfoPtrs = m_cvt.getConvertTypeInfoPtrs(typeId);
      storeObject = m_damagePolicy.eventDamagePolicy(damage, typeId.id());
    }        

    for (unsigned idx=0; idx < convertTypeInfoPtrs.size(); ++idx) {
      (*damageMap)[PSEvt::EventKey( convertTypeInfoPtrs[idx], xtc->src, "") ] = damage;
    }

    if (not storeObject) {
      MsgLog(name(),debug, name() << " damage = " << damage.value() 
             << " src=" << xtc->src << " not storing in Event. SmallDataProxy=" << isSmallDataProxy);
      continue;
    }

    boost::shared_ptr<Pds::Xtc> xptr(dgptr, xtc);   
    if (isSmallDataProxy) {
      if (not smallDataProxy) {
        MsgLog(name(), warning, name() << " smallDataProxy typeid found but smallDataProxy is null. Skiping (is this a .smd.xtc file?).");
      } else {
        smallDataProxy->addEventProxy(xptr, convertTypeInfoPtrs);
      }
    } else {
      m_cvt.convert(xptr, evt, env.configStore());
    }
    
  }

  if (smallDataProxy) smallDataProxy->finalize();

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

/// Fill environment from list of datagrams 
/// Datagrams should be sorted with DAQ streams first, Control streams last.
/// If transition is Configure or BeginCalibCycle, uses first DAQ stream 
/// (skips others) and all control streams.
void 
XtcInputModuleBase::fillEnv(const std::vector<XtcInput::Dgram> & dgList, Env& env)
{
  // If the same EventKey is in both the DAQ and Control stream,
  // we want the DAQ stream to take precedence.
  // we process the Control streams first, then the DAQ streams.
  // The DAQ entry will overwrite the Control entry. 

  // If this is a Configure or BeginCalibCycle, then all DAQ dgrams are *almost* entirely
  // duplicates of one another. The only difference observed is that some src's can include 
  // an IP address that will differ between the DAQ streams. These different src's will 
  // create different entries in the configStore (even though the data is the same).
  // It is cleaner and less error prone to only process one of the DAQ streams because of this case.

  // store control streams
  int numControlStored = 0;
  BOOST_REVERSE_FOREACH(const XtcInput::Dgram& dg, dgList) {
    bool isDaq = int(dg.file().stream()) < m_firstControlStream;
    if (isDaq) break;
    fillEnv(dg, env);
    ++numControlStored;
  }

  // store all Daq, or just first if config or beginCalibCycle
  int numDaqStored = 0;
  bool storedDaq = false;
  BOOST_FOREACH(const XtcInput::Dgram& dg, dgList) {
    bool isDaq = int(dg.file().stream()) < m_firstControlStream;
    if (isDaq and storedDaq and ::isConfigOrBeginCalib(dg)) break;
    if (not isDaq) break; // we already stored control streams
    fillEnv(dg, env);
    ++numDaqStored;
    if (isDaq) storedDaq = true;
  }
  MsgLog(name(), debug, name() << ": in fillEnv() from dgList of " 
         << dgList.size() << " dgrams. Stored "
         << numControlStored << " control dgrams and "
         << numDaqStored << " DAQ dgrams");
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

  boost::shared_ptr<psddl_pds2psana::SmallDataProxy> smallDataProxy = \
    psddl_pds2psana::SmallDataProxy::makeSmallDataProxy(dg.file(), m_liveMode, 
                                                        m_liveTimeOut, m_cvt, 0, env);

  if (::isConfigOrBeginCalib(dg)) {

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
            MsgLog(name(), warning, name() << ": failed to find Alias::ConfigV1 in config store");
          }
        }
      }
    }
    
 
    // Loop over all XTC contained in the datagram
    XtcInput::XtcIterator iter(&dgptr->xtc);
    while (Pds::Xtc* xtc = iter.next()) {
      if (xtc->contains.id() != Pds::TypeId::Id_Epics) {
        boost::shared_ptr<Pds::Xtc> xptr(dgptr, xtc);
        if (psddl_pds2psana::SmallDataProxy::isSmallDataProxy(xtc->contains)) {
          if (smallDataProxy) {
            // not checking if epics as already checked above
            smallDataProxy->addEnvProxy(xptr, psddl_pds2psana::SmallDataProxy::getSmallConvertTypeInfoPtrs(xtc, m_cvt));
          } else {
            MsgLog(name(), warning, name() << ": smallDataProxy typeid found but smallDataProxy is null. Skiping (is this a .smd.xtc file?).");
          }
        } else {
          // call the converter which will fill config store
          m_cvt.convertConfig(xptr, env.configStore());
        }
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
  if (smallDataProxy) smallDataProxy->finalize();

}


} // namespace PSXtcInput
