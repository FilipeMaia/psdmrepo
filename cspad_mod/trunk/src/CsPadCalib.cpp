//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadCalib...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "cspad_mod/CsPadCalib.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "cspad_mod/CalibDataProxy.h"
#include "cspad_mod/DataProxyMini.h"
#include "cspad_mod/DataProxyT.h"
#include "MsgLogger/MsgLogger.h"
#include "pdscalibdata/CsPadCommonModeSubV1.h"
#include "pdscalibdata/CsPadMiniPedestalsV1.h"
#include "pdscalibdata/CsPadMiniPixelStatusV1.h"
#include "pdscalibdata/CsPadPedestalsV1.h"
#include "pdscalibdata/CsPadPixelStatusV1.h"
#include "PSCalib/CalibFileFinder.h"
#include "psddl_psana/cspad.ddl.h"
#include "psddl_psana/cspad2x2.ddl.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace cspad_mod;
PSANA_MODULE_FACTORY(CsPadCalib)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace cspad_mod {

//----------------
// Constructors --
//----------------
CsPadCalib::CsPadCalib (const std::string& name)
  : Module(name)
  , m_inkey()
  , m_outkey()
  , m_doPedestals()
  , m_doPixelStatus()
  , m_doCommonMode()
{
  m_inkey = configStr("inputKey", "");
  m_outkey = configStr("outputKey", "calibrated");
  m_doPedestals = config("doPedestals", true);
  m_doPixelStatus = config("doPixelStatus", true);
  m_doCommonMode = config("doCommonMode", true);
}

//--------------
// Destructor --
//--------------
CsPadCalib::~CsPadCalib ()
{
}

/// Method which is called once at the beginning of the job
void 
CsPadCalib::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
CsPadCalib::beginRun(Event& evt, Env& env)
{
  // get run number
  shared_ptr<EventId> eventId = evt.get();
  int run = 0;
  if (eventId.get()) {
    run = eventId->run();
  } else {
    MsgLog(name(), warning, name() << ": Cannot determine run number, will use 0.");
  }

  // add proxies to calib store
  PSEnv::EnvObjectStore& calibStore = env.calibStore();

  if (m_doPedestals) {
    boost::shared_ptr< PSEvt::Proxy<pdscalibdata::CsPadMiniPedestalsV1> > proxy1(
        new CalibDataProxy<pdscalibdata::CsPadMiniPedestalsV1>(env.calibDir(), "CsPad::CalibV1", "pedestals", run));
    calibStore.putProxy(proxy1, PSEvt::EventKey::anySource());

    boost::shared_ptr< PSEvt::Proxy<pdscalibdata::CsPadPedestalsV1> > proxy2(
        new CalibDataProxy<pdscalibdata::CsPadPedestalsV1>(env.calibDir(), "CsPad::CalibV1", "pedestals", run));
    calibStore.putProxy(proxy2, PSEvt::EventKey::anySource());
  }

  if (m_doPixelStatus) {
    boost::shared_ptr< PSEvt::Proxy<pdscalibdata::CsPadMiniPixelStatusV1> > proxy1(
        new CalibDataProxy<pdscalibdata::CsPadMiniPixelStatusV1>(env.calibDir(), "CsPad::CalibV1", "pixel_status", run));
    calibStore.putProxy(proxy1, PSEvt::EventKey::anySource());

    boost::shared_ptr< PSEvt::Proxy<pdscalibdata::CsPadPixelStatusV1> > proxy2(
        new CalibDataProxy<pdscalibdata::CsPadPixelStatusV1>(env.calibDir(), "CsPad::CalibV1", "pixel_status", run));
    calibStore.putProxy(proxy2, PSEvt::EventKey::anySource());
  }

  if (m_doCommonMode) {
    boost::shared_ptr< PSEvt::Proxy<pdscalibdata::CsPadCommonModeSubV1> > proxy(
        new CalibDataProxy<pdscalibdata::CsPadCommonModeSubV1>(env.calibDir(), "CsPad::CalibV1", "common_mode", run));
    calibStore.putProxy(proxy, PSEvt::EventKey::anySource());
  }

}

/// Method which is called at the beginning of the calibration cycle
void 
CsPadCalib::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CsPadCalib::event(Event& evt, Env& env)
{

  // loop over all objects in event and find CsPad stuff
  const std::list<PSEvt::EventKey>& keys = evt.keys();
  for (std::list<PSEvt::EventKey>::const_iterator it = keys.begin(); it != keys.end(); ++ it) {

    const PSEvt::EventKey& key = *it;
    if (key.key() != m_inkey) continue;

    if (*key.typeinfo() == typeid(Psana::CsPad::DataV1)) {
      MsgLog(name(), debug, name() << ": found Psana::CsPad::DataV1 " << key);

      addProxyV1(key, evt, env);
      
    } else if (*key.typeinfo() == typeid(Psana::CsPad::DataV2)) {
      MsgLog(name(), debug, name() << ": found Psana::CsPad::DataV2 " << key);

      addProxyV2(key, evt, env);

    } else if (*key.typeinfo() == typeid(Psana::CsPad2x2::ElementV1)) {
      MsgLog(name(), debug, name() << ": found Psana::CsPad2x2::ElementV1 " << key);

      addProxyMini(key, evt, env);

    }

  }

}
  
/// Method which is called at the end of the calibration cycle
void 
CsPadCalib::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CsPadCalib::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CsPadCalib::endJob(Event& evt, Env& env)
{
}

void
CsPadCalib::addProxyV1(const PSEvt::EventKey& key, Event& evt, Env& env)
{
  // need an access to calib store
  PSEnv::EnvObjectStore& calibStore = env.calibStore();

  boost::shared_ptr< PSEvt::Proxy<Psana::CsPad::DataV1> > proxy(new DataProxyV1(key, calibStore));
  evt.putProxy(proxy, key.src(), m_outkey);
}

void
CsPadCalib::addProxyV2(const PSEvt::EventKey& key, Event& evt, Env& env)
{
  // need an access to calib store
  PSEnv::EnvObjectStore& calibStore = env.calibStore();

  boost::shared_ptr< PSEvt::Proxy<Psana::CsPad::DataV2> > proxy(new DataProxyV2(key, calibStore));
  evt.putProxy(proxy, key.src(), m_outkey);
}

void
CsPadCalib::addProxyMini(const PSEvt::EventKey& key, Event& evt, Env& env)
{
  // need an access to calib store
  PSEnv::EnvObjectStore& calibStore = env.calibStore();

  boost::shared_ptr< PSEvt::Proxy<Psana::CsPad2x2::ElementV1> > proxy(new DataProxyMini(key, calibStore));
  evt.putProxy(proxy, key.src(), m_outkey);
}

} // namespace cspad_mod
