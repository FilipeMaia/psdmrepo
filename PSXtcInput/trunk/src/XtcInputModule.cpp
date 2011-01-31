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
#include "PSXtcInput/Exceptions.h"
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
  unsigned dgSizeMB = config("dgSizeMB", 16);
  double l1offset = config("l1offset", 0.0);
  XtcStreamMerger::MergeMode merge = XtcStreamMerger::mergeMode(configStr("mergeMode", "FileName"));
  m_readerThread.reset( new boost::thread( DgramReader ( 
      files, *m_dgQueue, dgSizeMB*1048576, merge, false, l1offset) ) );
  
  
  // try to read first event and see if it is a Configure transition
  boost::shared_ptr<Pds::Dgram> dg(m_dgQueue->pop());
  if (not dg.get()) {
    // Nothing there at all, this is unexpected
    throw EmptyInput(ERR_LOC);
  }
  
  if ( dg->seq.service() != Pds::TransitionId::Configure ) {
    // Something else than Configure, store if for event()
    MsgLog(logger, warning, "Expected Configure transition for first datagram, received " 
           << Pds::TransitionId::name(dg->seq.service()) );
    m_putBack = dg;
    return;
  }
  
  // Store configuration info in the environment
  fillEnv(dg, env);

}

InputModule::Status 
XtcInputModule::event(Event& evt, Env& env)
{
  // get datagram either from saved event or queue
  boost::shared_ptr<Pds::Dgram> dg;
  if (m_putBack.get()) {
    dg = m_putBack;
    m_putBack.reset();
  } else {
    dg.reset(m_dgQueue->pop());
  }

  if (not dg.get()) {
    // finita
    return Stop;
  }

  Status status = Skip;
  switch( dg->seq.service()) {
  
  case Pds::TransitionId::Configure:
    // That should not happen
    MsgLog(logger, error, "Unexpected Configure transition");
    status = Skip;
    break;
    
  case Pds::TransitionId::Unconfigure:
    // Nothing is expected after this
    status = Stop;
    break;
 
  case Pds::TransitionId::BeginRun:
    // signal new run, content is not relevant
    status = BeginRun;
    break;
  
  case Pds::TransitionId::EndRun:
    // signal end of run, content is not relevant
    status = EndRun;
    break;
  
  case Pds::TransitionId::BeginCalibCycle:
    // copy config data and signal new calib cycle
    fillEnv(dg, env);
    status = BeginCalibCycle;
    break;
  
  case Pds::TransitionId::EndCalibCycle:
    // stop calib cycle
    status = EndCalibCycle;
    break;
  
  case Pds::TransitionId::L1Accept:
    // regular event
    fillEnv(dg, env);
    fillEvent(dg, evt);
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
    status = Skip;
    break;
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
XtcInputModule::fillEvent(const boost::shared_ptr<Pds::Dgram>& dg, Event& evt)
{
  // Store datagram itself in the event
  evt.put(dg);
  
  const Pds::Sequence& seq = dg->seq ;
  const Pds::ClockTime& clock = seq.clock() ;
  std::cout << Pds::TransitionId::name(seq.service()) << " transition: damage " << dg->xtc.damage.value() 
        << ", type " << int(seq.type()) 
        << ", time " << clock.seconds() << " sec " << clock.nanoseconds() << " nsec"
        << ", payloadSize " << dg->xtc.sizeofPayload() << "\n";

  XtcInput::XtcIterator iter(&dg->xtc);
  while (Pds::Xtc* xtc = iter.next()) {
    
    std::cout << "  XTC type " << Pds::TypeId::name(xtc->contains.id())
          << " V" << xtc->contains.version()
          << ", payloadSize " << xtc->sizeofPayload() << "\n";
    
    
  }
}

// Fill environment with datagram contents
void 
XtcInputModule::fillEnv(const boost::shared_ptr<Pds::Dgram>& dg, Env& env)
{
  // All objects in datagram in Configuration and BeginCalibCycle transitions
  // (except for EPICS data) are considered configuration data. Just store them
  // them in the ConfigStore part of the environment

}

} // namespace PSXtcInput
