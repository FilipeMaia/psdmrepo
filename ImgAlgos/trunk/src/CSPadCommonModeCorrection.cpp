//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadCommonModeCorrection...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadCommonModeCorrection.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
//#include "psddl_psana/acqiris.ddl.h"
#include "psddl_psana/cspad2x2.ddl.h"

#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace Psana;
using namespace ImgAlgos;

// This declares this class as psana module

PSANA_MODULE_FACTORY(CSPadCommonModeCorrection)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
CSPadCommonModeCorrection::CSPadCommonModeCorrection (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_src()
  , m_inkey()
  , m_outkey()
  , m_maxEvents()
  , m_ampThr()
  , m_filter()
  , m_segMask()
  , m_count(0)
 {
  // get the values from configuration or use defaults
  m_str_src    = configStr("source", "DetInfo(:Cspad)"); // "DetInfo()", "CxiDs1.0:Cspad.0", "CxiSc1.0:Cspad2x2.0"
  m_inkey      = configStr("inputKey", "calibrated");
  m_outkey     = configStr("outputKey", "");
  m_maxEvents  = config   ("events", 20U); // 1<<31U
  m_ampThr     = config   ("ampthr", 30);
  m_filter     = config   ("filter", false);

   // initialize arrays
  std::fill_n(&m_segMask[0], int(MaxQuads), 0U);
 }

//--------------
// Destructor --
//--------------
CSPadCommonModeCorrection::~CSPadCommonModeCorrection ()
{
}

/// Method which is called once at the beginning of the job
void 
CSPadCommonModeCorrection::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
CSPadCommonModeCorrection::beginRun(Event& evt, Env& env)
{
  // Find all configuration objects matching the source address
  // provided in configuration. If there is more than one configuration 
  // object is found then complain and stop.
  
  std::string src = m_str_src;

  int count = 0;

  // need to know segment mask which is availabale in configuration only
  shared_ptr<Psana::CsPad::ConfigV1> config1 = env.configStore().get(src, &m_src);
  if (config1.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config1->asicMask()==1 ? 0x3 : 0xff; }
    ++ count;
  }
  
  shared_ptr<Psana::CsPad::ConfigV2> config2 = env.configStore().get(src, &m_src);
  if (config2.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config2->roiMask(i); }
    ++ count;
  }

  shared_ptr<Psana::CsPad::ConfigV3> config3 = env.configStore().get(src, &m_src);
  if (config3.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config3->roiMask(i); }
    ++ count;
  }

  if (not count) {
    MsgLog(name(), error, "No CsPad configuration objects found, terminating.");
    terminate();
    return;
  }
  
  if (count > 1) {
    MsgLog(name(), error, "Multiple CsPad configuration objects found, use more specific source address. Terminating.");
    terminate();
    return;
  }

  WithMsgLog(name(), info, log) {
      log <<  "Found CsPad object with address " << m_src << " with masks:";
      for (int i = 0; i < MaxQuads; ++i) log << " " << m_segMask[i];
  } 

  const Pds::DetInfo& dinfo = static_cast<const Pds::DetInfo&>(m_src);
  // validate that this is indeed cspad, should always be true, but
  // additional protection here should not hurt
  if (dinfo.device() != Pds::DetInfo::Cspad) {
    MsgLog(name(), error, "Found Cspad configuration object with invalid address. Terminating.");
    terminate();
    return;
  }
}

/// Method which is called at the beginning of the calibration cycle
void 
CSPadCommonModeCorrection::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadCommonModeCorrection::event(Event& evt, Env& env)
{
  // example of getting non-detector data from event
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    MsgLog(name(), debug, "event ID: " << *eventId);
  }
  
  // this is how to skip event (all downstream modules will not be called)
  if (m_filter && m_count % 10 == 0) skip();
  
  // this is how to gracefully stop analysis job
  if (m_count >= m_maxEvents) stop();

  //-------------------- Time
  struct timespec start, stop;
  int status = clock_gettime( CLOCK_REALTIME, &start ); // Get LOCAL time
  //-------------------- Time


  // loop over all objects in event and find CsPad stuff
  const std::list<PSEvt::EventKey>& keys = evt.keys();
  for (std::list<PSEvt::EventKey>::const_iterator it = keys.begin(); it != keys.end(); ++ it) {

    const PSEvt::EventKey& key = *it;

    if (key.key() != m_inkey) continue;
    MsgLog(name(), debug, "Process event for key =" << key.key() );    

    getAndProcessDataset(evt, env, key.key());
  }

  //-------------------- Time
  status = clock_gettime( CLOCK_REALTIME, &stop ); // Get LOCAL time
  cout << "  Event: " << m_count 
       << "  Time to process one event is " 
       << stop.tv_sec - start.tv_sec + 1e-9*(stop.tv_nsec - start.tv_nsec) 
       << " sec" << endl;
  //-------------------- Time

}

/// A part of the event method
void 
CSPadCommonModeCorrection::getAndProcessDataset(Event& evt, Env& env, const std::string& key)
{
    shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(m_src, key, &m_actualSrc);
    if (data1.get()) {
  
      ++ m_count;
      
      int nQuads = data1->quads_shape()[0];
      for (int iq = 0; iq != nQuads; ++ iq) {
        
        const CsPad::ElementV1& quad = data1->quads(iq); // get quad object
        // process event for this quad
        const ndarray<int16_t, 3>& data = quad.data();
        processQuad(quad.quad(), data.data());
      }      
    }

    shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_src, key, &m_actualSrc);
    if (data2.get()) {
  
      ++ m_count;
      
      int nQuads = data2->quads_shape()[0];
      for (int iq = 0; iq != nQuads; ++ iq) {
        
        const CsPad::ElementV2& quad = data2->quads(iq); // get quad object  
        // process event for this quad
        const ndarray<int16_t, 3>& data = quad.data();
        processQuad(quad.quad(), data.data());
      }
    }
}
  
/// Method which is called at the end of the calibration cycle
void 
CSPadCommonModeCorrection::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadCommonModeCorrection::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadCommonModeCorrection::endJob(Event& evt, Env& env)
{
}

/// process event for quad
void 
CSPadCommonModeCorrection::processQuad(unsigned qNum, const int16_t* data)
{
  // loop over segments
  int seg = 0;
  for (int sect = 0; sect < MaxSectors; ++ sect) {
    if (m_segMask[qNum] & (1 << sect)) {

      double sum  = 0;
      int    npix = 0;

      const int16_t* segData = data + seg*SectorSize;

      // evaluate the common mode amplitude as an averaged over pixels below threshold
      for (int i = 0; i < SectorSize; ++ i) { 
	if (segData[i] > m_ampThr ) continue;
          sum+= double(segData[i]); 
          ++ npix;
      }                

      int16_t average = (npix>0) ? int16_t(sum/npix) : 0 ;
      //MsgLog(name(), info, "   sect: " << sect << "  sum = " << sum  << "  npix = " << npix << "  average = " << average  );          
      int16_t* corrData = &m_data_corr[qNum][0][0][0] + seg*SectorSize;

      // Apply the common mode correction
      for (int i = 0; i < SectorSize; ++ i) { 
	corrData[i] = segData[i] - average;
      }                
      ++seg;
    }
  }
}

} // namespace ImgAlgos
