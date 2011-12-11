//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadFilter...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "cspad_mod/CsPadFilter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <numeric>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "cspad_mod/CalibDataProxy.h"
#include "MsgLogger/MsgLogger.h"
#include "pdscalibdata/CsPadFilterV1.h"
#include "PSCalib/CalibFileFinder.h"
#include "psddl_psana/cspad.ddl.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace cspad_mod;
PSANA_MODULE_FACTORY(CsPadFilter)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace cspad_mod {

//----------------
// Constructors --
//----------------
CsPadFilter::CsPadFilter (const std::string& name)
  : Module(name)
  , m_src()
  , m_key()
  , m_skipIfNoData()
  , m_mode()
  , m_param()
{
  m_src = configStr("source", "DetInfo(:Cspad)");
  m_key = configStr("inputKey", "");
  m_skipIfNoData = config("skipIfNoData", true);
  m_mode = config("mode", -1);
  m_param = configList("parameters", std::list<double>());
}

//--------------
// Destructor --
//--------------
CsPadFilter::~CsPadFilter ()
{
}

/// Method which is called once at the beginning of the job
void 
CsPadFilter::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
CsPadFilter::beginRun(Event& evt, Env& env)
{
  if (m_mode<= 0) {
    
    // use calibration file
    
    // get run number
    shared_ptr<EventId> eventId = evt.get();
    int run = 0;
    if (eventId.get()) {
      run = eventId->run();
      MsgLog(name(), trace, name() << ": Using run number " << run);
    } else {
      MsgLog(name(), warning, name() << ": Cannot determine run number, will use 0.");
    }
  
    // add proxy to calib store
    boost::shared_ptr< PSEvt::Proxy<pdscalibdata::CsPadFilterV1> > proxy(
        new CalibDataProxy<pdscalibdata::CsPadFilterV1>(env.calibDir(), "CsPad::CalibV1", "filter", run));
    env.calibStore().putProxy(proxy, PSEvt::EventKey::anySource());

  } else {
    
    // copy parameters to array
    const int paramSize = pdscalibdata::CsPadFilterV1::DataSize;
    double param[paramSize];    
    std::fill_n(param, paramSize, 0.0);
    
    std::list<double>::const_iterator it = m_param.begin();
    for (int i = 0; i != paramSize and it != m_param.end(); ++ i, ++ it) {
      param[i] = *it;
    }
    if (it != m_param.end()) {
      MsgLog(name(), warning, "Too many values specified in parameters, ignoring extra values.");
    }

    // make filter object
    pdscalibdata::CsPadFilterV1::FilterMode mode = pdscalibdata::CsPadFilterV1::FilterMode(m_mode);
    boost::shared_ptr<pdscalibdata::CsPadFilterV1> filter(new pdscalibdata::CsPadFilterV1(mode, param));
    
    // store it
    env.calibStore().put(filter, PSEvt::EventKey::anySource());
  }
}

/// Method which is called at the beginning of the calibration cycle
void 
CsPadFilter::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CsPadFilter::event(Event& evt, Env& env)
{

  Pds::Src actualSrc;

  shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(m_src, m_key, &actualSrc);
  if (data1.get()) {

    // get calibration object
    boost::shared_ptr<pdscalibdata::CsPadFilterV1> filter = env.calibStore().get(actualSrc);
    if (filter.get()) {

      // loop over all quads
      int nQuads = data1->quads_shape()[0];
      for (int q = 0; q < nQuads; ++ q) {

        const Psana::CsPad::ElementV1& el = data1->quads(q);

        // call filter
        bool stat = filter->filter(el.data());
        if (stat) {
          // at least some data is good, do not skip and stop here
          MsgLog(name(), debug, name() << ": Good data found in CsPad::DataV1 quadrant=" << el.quad());
          return;
        }

      }

      // no good data found, skip it
      MsgLog(name(), debug, name() << ": No good data found in CsPad::DataV1");
      skip();
      return;

    }
  }


  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_src, m_key, &actualSrc);
  if (data2.get()) {

    // get calibration object
    boost::shared_ptr<pdscalibdata::CsPadFilterV1> filter = env.calibStore().get(actualSrc);
    if (filter.get()) {

      // loop over all quads
      int nQuads = data2->quads_shape()[0];
      for (int q = 0; q < nQuads; ++ q) {

        const Psana::CsPad::ElementV2& el = data2->quads(q);

        // call filter
        bool stat = filter->filter(el.data());
        if (stat) {
          // at least some data is good, do not skip and stop here
          MsgLog(name(), debug, name() << ": Good data found in CsPad::DataV2 quadrant=" << el.quad());
          return;
        }

      }

      // no good data found, skip it
      MsgLog(name(), debug, name() << ": No good data were found in CsPad::DataV2");
      skip();
      return;

    }
  }


  shared_ptr<Psana::CsPad::MiniElementV1> mini1 = evt.get(m_src, m_key, &actualSrc);
  if (mini1.get()) {

    // get calibration object
    boost::shared_ptr<pdscalibdata::CsPadFilterV1> filter = env.calibStore().get(actualSrc);
    if (filter.get()) {

      const Psana::CsPad::MiniElementV1& el = *mini1;

      // call filter
      bool stat = filter->filter(el.data());
      if (stat) {
        // at least some data is good, do not skip and stop here
        MsgLog(name(), debug, name() << ": Good data found in CsPad::MiniElementV1 quadrant=" << el.quad());
        return;
      }

      // no good data found, skip it
      MsgLog(name(), debug, name() << ": No good data were found in CsPad::MiniElementV1");
      skip();
      return;

    }
  }

  // no data found
  MsgLog(name(), debug, name() << ": No cspad data was found in event");
  if (m_skipIfNoData) {
    skip();
  }

}
  
/// Method which is called at the end of the calibration cycle
void 
CsPadFilter::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CsPadFilter::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CsPadFilter::endJob(Event& evt, Env& env)
{
}

} // namespace cspad_mod
