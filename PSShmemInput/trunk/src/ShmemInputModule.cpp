//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ShmemInputModule...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSShmemInput/ShmemInputModule.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "IData/Dataset.h"
#include "MsgLogger/MsgLogger.h"
#include "PSShmemInput/Exceptions.h"
#include "PSShmemInput/ShmemMonitorClient.h"
#include "XtcInput/DgramQueue.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace PSShmemInput;
PSANA_INPUT_MODULE_FACTORY(ShmemInputModule)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSShmemInput {

//----------------
// Constructors --
//----------------
ShmemInputModule::ShmemInputModule(const std::string& name)
  : PSXtcInput::XtcInputModuleBase(name)
  , m_dgQueue(new XtcInput::DgramQueue(32))
  , m_readerThread()
{
}

//--------------
// Destructor --
//--------------
ShmemInputModule::~ShmemInputModule ()
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

// Initialization method for external datagram source
void
ShmemInputModule::initDgramSource()
{
  // get input spec
  std::vector<std::string> inputs = configList("input");
  if (inputs.empty()) throw EmptyInputList(ERR_LOC);
  if (inputs.size() > 1) throw DatasetSpecError(ERR_LOC, "more than one dataset", configStr("input"));

  // make dataset out of it
  IData::Dataset ds(inputs[0]);

  // need "shmem=name.N" spec
  if (not ds.exists("shmem")) throw DatasetSpecError(ERR_LOC, "'shmem' key is missing", inputs[0]);

  std::string shmem = ds.value("shmem");
  std::string::size_type p = shmem.rfind('.');
  if (p == std::string::npos) throw DatasetSpecError(ERR_LOC, "comma missing in shmem value", inputs[0]);

  std::string tag(shmem, 0, p);
  int index = 0;
  try {
    index = boost::lexical_cast<int>(shmem.substr(p+1));
  } catch (const boost::bad_lexical_cast& ex) {
    throw DatasetSpecError(ERR_LOC, "bad index in shmem value", inputs[0]);
  }

  // stop transition
  Pds::TransitionId::Value stopTr = Pds::TransitionId::EndRun;
  if (ds.exists("stop")) {
    std::string stopTrStr = ds.value("stop");
    boost::algorithm::to_lower(stopTrStr);

    if (stopTrStr == "unmap") {
      stopTr = Pds::TransitionId::Unmap;
    } else if (stopTrStr == "unconfigure") {
      stopTr = Pds::TransitionId::Unconfigure;
    } else if (stopTrStr == "endrun") {
      stopTr = Pds::TransitionId::EndRun;
    } else if (stopTrStr == "endcalibcycle") {
      stopTr = Pds::TransitionId::EndCalibCycle;
    } else if (stopTrStr == "none" or stopTrStr == "no") {
      // this transition cannot happen, means non-stop running
      stopTr = Pds::TransitionId::NumberOf;
    } else {
      throw DatasetSpecError(ERR_LOC, "unexpected stop condition", inputs[0]);
    }
  }

  m_readerThread.reset(new boost::thread(ShmemMonitorClient(tag, index, *m_dgQueue, stopTr)));
}

// Get the next datagram from some external source
XtcInput::Dgram
ShmemInputModule::nextDgram()
{
  return m_dgQueue->pop();
}

} // namespace PSShmemInput
