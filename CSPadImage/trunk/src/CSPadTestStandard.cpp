//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadTestStandard...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadImage/CSPadTestStandard.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
#include "psddl_psana/acqiris.ddl.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace CSPadImage;
PSANA_MODULE_FACTORY(CSPadTestStandard)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadImage {

//----------------
// Constructors --
//----------------
CSPadTestStandard::CSPadTestStandard (const std::string& name)
  : Module(name)
  , m_src()
  , m_maxEvents()
  , m_filter()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_src = configStr("source", "DetInfo(:Acqiris)");
  m_maxEvents = config("events", 32U);
  m_filter = config("filter", false);
}

//--------------
// Destructor --
//--------------
CSPadTestStandard::~CSPadTestStandard ()
{
}

/// Method which is called once at the beginning of the job
void 
CSPadTestStandard::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
CSPadTestStandard::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
CSPadTestStandard::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadTestStandard::event(Event& evt, Env& env)
{
  // example of getting non-detector data from event
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    // example of producing messages using MgsLog facility
    MsgLog(name(), info, "event ID: " << *eventId);
  }
  
  // tis is how to skip event (all downstream modules will not be called)
  if (m_filter && m_count % 10 == 0) skip();
  
  // this is how to gracefully stop analysis job
  if (m_count >= m_maxEvents) stop();

  // example of getting detector data from event
  shared_ptr<Psana::Acqiris::DataDescV1> acqData = evt.get(m_src);
  if (acqData.get()) {
    // another example of printing using MgsLog facility
    WithMsgLog(name(), debug, log) {
      log << "found Acqiris::DataDescV1 object";
    }
  }
  
  // increment event counter
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
CSPadTestStandard::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadTestStandard::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadTestStandard::endJob(Event& evt, Env& env)
{
}

} // namespace CSPadImage
