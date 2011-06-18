//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcInputModule...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcInput/XtcInputModule.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <iterator>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSTime/Time.h"
#include "PSXtcInput/Exceptions.h"
#include "PSXtcInput/XtcEventId.h"
#include "XtcInput/DgramQueue.h"
#include "XtcInput/DgramReader.h"
#include "XtcInput/XtcFileName.h"
#include "XtcInput/XtcIterator.h"
#include "XtcInput/XtcStreamMerger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace XtcInput;

using namespace PSXtcInput;
PSANA_INPUT_MODULE_FACTORY(XtcInputModule)

namespace {
  
  const char logger[] = "XtcInputModule";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcInput {

//----------------
// Constructors --
//----------------
XtcInputModule::XtcInputModule (const std::string& name)
  : InputModule(name)
  , m_dgQueue(new XtcInput::DgramQueue(10))
  , m_putBack()
  , m_readerThread()
  , m_cvt()
{
}

//--------------
// Destructor --
//--------------
XtcInputModule::~XtcInputModule ()
{
  if (m_readerThread) {
    // ask the thread to stop
    m_readerThread->interrupt();
    MsgLog(logger, debug, "wait for reader thread to finish");
    // wait until it does
    m_readerThread->join();
    MsgLog(logger, debug, "reader thread has finished");
  }
}

/// Method which is called once at the beginning of the job
void 
XtcInputModule::beginJob(Env& env)
{
  MsgLog(logger, debug, "XtcInputModule: in beginJob()");

  // will throw if no files were defined in config
  std::list<std::string> fileNames = configList("files");
  if ( fileNames.empty() ) {
    throw EmptyFileList(ERR_LOC);
  }
  
  std::list<XtcInput::XtcFileName> files ;
  for (std::list<std::string>::const_iterator i = fileNames.begin(); i != fileNames.end(); ++i) {
    files.push_back( XtcInput::XtcFileName(*i) ) ;
  }
  WithMsgLog(logger, debug, str) {
    str << "Input files: ";
    std::copy(files.begin(), files.end(), std::ostream_iterator<XtcInput::XtcFileName>(str, " "));
  }
  
  // start reader thread
  unsigned dgSizeMB = config("dgSizeMB", 32);
  double l1offset = config("l1offset", 0.0);
  XtcStreamMerger::MergeMode merge = XtcStreamMerger::mergeMode(configStr("mergeMode", "FileName"));
  m_readerThread.reset( new boost::thread( DgramReader ( 
      files, *m_dgQueue, dgSizeMB*1048576, merge, false, l1offset) ) );
  
  // try to read first event and see if it is a Configure transition
  XtcInput::Dgram dg(m_dgQueue->pop());
  if (dg.empty()) {
    // Nothing there at all, this is unexpected
    throw EmptyInput(ERR_LOC);
  }
  
  Dgram::ptr dgptr = dg.dg();

  MsgLog(logger, debug, "XtcInputModule: read first datagram, transition = " 
        << Pds::TransitionId::name(dgptr->seq.service()));

  if ( dgptr->seq.service() != Pds::TransitionId::Configure ) {
    // Something else than Configure, store if for event()
    MsgLog(logger, warning, "Expected Configure transition for first datagram, received " 
           << Pds::TransitionId::name(dgptr->seq.service()) );
    m_putBack = dg;
    return;
  }
  
  m_transitions[dgptr->seq.service()] = dgptr->seq.clock();
  
  // Store configuration info in the environment
  fillEnv(dg, env);

}

InputModule::Status 
XtcInputModule::event(Event& evt, Env& env)
{
  MsgLog(logger, debug, "XtcInputModule: in event()");

  Status status = Skip;
  bool found = false;
  while (not found) {

    // get datagram either from saved event or queue
    XtcInput::Dgram dg;
    if (not m_putBack.empty()) {
      dg = m_putBack;
      m_putBack = Dgram();
    } else {
      dg = m_dgQueue->pop();
    }
  
    if (dg.empty()) {
      // finita
      MsgLog(logger, debug, "EOF seen");
      return Stop;
    }


    const Pds::Sequence& seq = dg.dg()->seq ;
    const Pds::ClockTime& clock = seq.clock() ;
    Pds::TransitionId::Value trans = seq.service();

    MsgLog(logger, debug, "XtcInputModule: found new datagram, transition = " 
          << Pds::TransitionId::name(trans));

    switch( seq.service()) {
    
    case Pds::TransitionId::Configure:
      if (not (clock == m_transitions[trans])) {
        MsgLog(logger, warning, "Multiple Configure transitions encountered");
        m_transitions[trans] = clock;
        fillEnv(dg, env);
      }
      break;
      
    case Pds::TransitionId::Unconfigure:
      break;
   
    case Pds::TransitionId::BeginRun:
      // signal new run, content is not relevant
      if (not (clock == m_transitions[trans])) {
        status = BeginRun;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::EndRun:
      // signal end of run, content is not relevant
      if (not (clock == m_transitions[trans])) {
        status = EndRun;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::BeginCalibCycle:
      // copy config data and signal new calib cycle
      if (not (clock == m_transitions[trans])) {
        fillEnv(dg, env);
        status = BeginCalibCycle;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::EndCalibCycle:
      // stop calib cycle
      if (not (clock == m_transitions[trans])) {
        status = EndCalibCycle;
        found = true;
        m_transitions[trans] = clock;
      }
      break;
    
    case Pds::TransitionId::L1Accept:
      // regular event
      fillEnv(dg, env);
      fillEvent(dg, evt, env);
      found = true;
      status = DoEvent;
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
XtcInputModule::endJob(Env& env)
{
}

// Fill event with datagram contents
void 
XtcInputModule::fillEvent(const XtcInput::Dgram& dg, Event& evt, Env& env)
{
  MsgLog(logger, debug, "XtcInputModule: in fillEvent()");

  Dgram::ptr dgptr = dg.dg();
    
  // Store datagram itself in the event
  evt.put(dgptr);
  
  const Pds::Sequence& seq = dgptr->seq ;
  const Pds::ClockTime& clock = seq.clock() ;

  // Store event ID
  PSTime::Time evtTime(clock.seconds(), clock.nanoseconds());
  unsigned run = dg.file().run();
  boost::shared_ptr<PSEvt::EventId> eventId( new XtcEventId(run, evtTime) );
  evt.put(eventId);

  // Loop over all XTC contained in the datagram
  XtcInput::XtcIterator iter(&dgptr->xtc);
  while (Pds::Xtc* xtc = iter.next()) {
      
    boost::shared_ptr<Pds::Xtc> xptr(dgptr, xtc);
    // call the converter which will fill event with data
    m_cvt.convert(xptr, evt, env.configStore());
    
  }
}

// Fill environment with datagram contents
void 
XtcInputModule::fillEnv(const XtcInput::Dgram& dg, Env& env)
{
  MsgLog(logger, debug, "XtcInputModule: in fillEnv()");

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
