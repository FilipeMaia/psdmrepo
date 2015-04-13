//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeTool::Setup...
//
// Author List:
//      Matthew J. Weaver
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TimeTool/Setup.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include "psalg/psalg.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
// to work with detector data include corresponding 
// header from psddl_psana package
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
#include "psddl_psana/evr.ddl.h"
#include "psddl_psana/lusi.ddl.h"
#include "psddl_psana/camera.ddl.h"
#include "psddl_psana/opal1k.ddl.h"

#include <iomanip>
#include <fstream>
#include <sstream>

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace TimeTool;
PSANA_MODULE_FACTORY(Setup)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace TimeTool {

//----------------
// Constructors --
//----------------
Setup::Setup (const std::string& name)
  : Module(name), m_hacf(0)
{
  // get the values from configuration or use defaults
  m_get_key = configSrc("get_key","DetInfo(:Opal1000)");

  m_eventcode_nobeam = config("eventcode_nobeam",162);

  m_eventcode_skip   = config("eventcode_skip");

  m_ipm_get_key        = configStr("ipm_get_key");
  m_ipm_beam_threshold = config("ipm_beam_threshold",0.);

  m_projectX  = config("projectX",true);
  m_proj_cut  = config("proj_cut" ,0);

#define initList(l)                             \
  std::vector<unsigned> l = configList(#l);

  initList(sig_roi_x)
  initList(sig_roi_y)
  initList(sb_roi_x)
  initList(sb_roi_y)
#undef initList

  m_frame_roi[0] = m_frame_roi[1] = 0;

  if (sb_roi_x.size() && sb_roi_y.size()) {
    if (m_projectX) {
      if (sb_roi_x[0]!=sig_roi_x[0] ||
          sb_roi_x[1]!=sig_roi_x[1]) {
        sb_roi_x[0] = sig_roi_x[0];
        sb_roi_x[1] = sig_roi_x[1];
        MsgLog(name, warning, 
               name << ": Signal and sideband roi x ranges differ.  Setting sideband roi to signal.");
      }
      if ((sb_roi_y[1]-sb_roi_y[0]) != (sig_roi_y[1]-sig_roi_y[0])) {
        MsgLog(name, fatal, 
               name << ": Signal and sideband roi y range sizes differ.");
      }
    }
    else {
      if (sb_roi_y[0]!=sig_roi_y[0] ||
          sb_roi_y[1]!=sig_roi_y[1]) {
        sb_roi_y[0] = sig_roi_y[0];
        sb_roi_y[1] = sig_roi_y[1];
        MsgLog(name, warning, 
               name << ": Signal and sideband roi y ranges differ.  Setting sideband roi to signal.");
      }
      if ((sb_roi_x[1]-sb_roi_x[0]) != (sig_roi_x[1]-sig_roi_x[0])) {
        MsgLog(name, fatal, 
               name << ": Signal and sideband roi x range sizes differ.");
      }
    }
    m_sb_roi_lo[0] = sb_roi_y[0];
    m_sb_roi_hi[0] = sb_roi_y[1];
    m_sb_roi_lo[1] = sb_roi_x[0];
    m_sb_roi_hi[1] = sb_roi_x[1];
  }
  else {
    m_sb_roi_lo[0] = m_sb_roi_hi[0] = 0;
    m_sb_roi_lo[1] = m_sb_roi_hi[1] = 0;
  }

  m_sig_roi_lo[0] = sig_roi_y[0];
  m_sig_roi_hi[0] = sig_roi_y[1];
  m_sig_roi_lo[1] = sig_roi_x[0];
  m_sig_roi_hi[1] = sig_roi_x[1];

  m_sb_avg_fraction  = config("sb_avg_fraction",0.05);
  m_ref_avg_fraction = config("ref_avg_fraction",1.);

  std::string ref_load  = configStr("ref_load");
  if (!ref_load.empty()) {
    std::ifstream s(ref_load.c_str());
    std::vector<double> ref;
    while(s.good()) {
      double v;
      s >> v;
      ref.push_back(v);
    }

    m_ref = make_ndarray<double>(ref.size());
    for(unsigned i=0; i<ref.size(); i++)
      m_ref[i] = ref[i];

    if (m_ref.size()==0) {
      MsgLog(name, fatal, 
             name << ": Reference load failed");
    }
  }
  else {
    MsgLog(name, fatal, 
           name << ": No ref_load parameter provided");
  }

  m_pedestal  = 0;

  unsigned ndump = config("dump",0);
  for(unsigned i=0; i<ndump; i++)
    m_hdump.push_back(DumpH());
}

//--------------
// Destructor --
//--------------
Setup::~Setup ()
{
}

/// Method which is called once at the beginning of the job
void 
Setup::beginJob(Event& evt, Env& env)
{
  {
    shared_ptr<Psana::Opal1k::ConfigV1> config = 
      env.configStore().get(m_get_key);
    if (config.get()) {
      m_pedestal = config->output_offset();
      MsgLog(name(), info, 
             name() << ": found configured offset of " << m_pedestal);
    }
  }

  {
    shared_ptr<Psana::Camera::FrameFexConfigV1> config = 
      env.configStore().get(m_get_key);
    if (config.get() && 
        config->forwarding()==Psana::Camera::FrameFexConfigV1::RegionOfInterest) {
      MsgLog(name(), info, 
             name() << ": found configured roi of [" 
             << config->roiBegin().column() << ','
             << config->roiEnd  ().column() << "] ["
             << config->roiBegin().row() << ','
             << config->roiEnd  ().row() << "]");

      //  current software does not correct for ROI
#if 0
      unsigned roi_y = config->roiBegin().row();
      m_frame_roi [0] = roi_y;
      m_sig_roi_lo[0] -= roi_y;
      m_sig_roi_hi[0] -= roi_y;
      m_sb_roi_lo [0] -= roi_y;
      m_sb_roi_hi [0] -= roi_y;

      unsigned roi_x = config->roiBegin().column();
      m_frame_roi [1] = roi_x;
      m_sig_roi_lo[1] -= roi_x;
      m_sig_roi_hi[1] -= roi_x;
      m_sb_roi_lo [1] -= roi_x;
      m_sb_roi_hi [1] -= roi_x;
#endif
    }
  }
  
  m_hmgr = env.hmgr();

  if (not m_hmgr and m_hdump.size()>0) {
    MsgLog(name(), error, "histogram manager returned by psana env "
           "is null, but hisogramming has been requested in configuration");
  }

  if (m_hmgr) {
    unsigned pdim = m_projectX ? 1:0;
    Axis a(m_sig_roi_hi[pdim]-m_sig_roi_lo[pdim]+1,
           double(m_sig_roi_lo[pdim])-0.5,double(m_sig_roi_hi[pdim]+0.5));
    for(unsigned i=0; i<m_hdump.size(); i++) {
      { std::stringstream s;
        s << "Raw projection: event " << i;
        m_hdump[i].hraw = m_hmgr->hist1d(s.str().c_str(),"projection",a); }
      { std::stringstream s;
        s << "Corr projection: event " << i;
        m_hdump[i].hcor = m_hmgr->hist1d(s.str().c_str(),"projection",a); }
      { std::stringstream s;
        s << "Corr reference: event " << i;
        m_hdump[i].href = m_hmgr->hist1d(s.str().c_str(),"projection",a); }
      { std::stringstream s;
        s << "Ratio: event " << i;
        m_hdump[i].hrat = m_hmgr->hist1d(s.str().c_str(),"ratio",a); }
      { std::stringstream s;
        s << "ACF: event " << i;
        m_hdump[i].hacf = m_hmgr->hist1d(s.str().c_str(),"acf",a); }
    }

    m_hacf = m_hmgr->prof1("Reference ACF","autocorrelation",a);
  } 
  m_count=0;
}

/// Method which is called at the beginning of the run
void 
Setup::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
Setup::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
Setup::event(Event& evt, Env& env)
{
  bool nobeam=false;
  { 
    //
    //  Beam is absent if BYKIK fired
    //
    shared_ptr<Psana::EvrData::DataV3> evr4 = evt.get(Source("DetInfo(:Evr)"));
    shared_ptr<Psana::EvrData::DataV3> evr3;
    if (not evr4) evr3 = evt.get(Source("DetInfo(:Evr)"));
    if (evr3 or evr4) {
      bool nolaser=false;
      ndarray<const Psana::EvrData::FIFOEvent,1> f = \
        evr4 ? evr4->fifoEvents() : evr3->fifoEvents();
      unsigned ec_nobeam = unsigned(abs(m_eventcode_nobeam));
      unsigned ec_skip   = unsigned(abs(m_eventcode_skip));
      for(unsigned i=0; i<f.shape()[0]; i++) {
        nobeam  |= (f[i].eventCode()==ec_nobeam);
        nolaser |= (f[i].eventCode()==ec_skip);
        MsgLog(name(), trace, 
               name() << ": Found EVR code " << f[i].eventCode());
      }

      if (m_eventcode_nobeam  < 0) nobeam  = !nobeam;
      if (m_eventcode_skip    < 0) nolaser = !nolaser;

      if (nolaser) return;
    }
    else {
      MsgLog(name(), warning, 
             name() << ": Failed to retrieve EVR event data - tried DataV3 and DataV4.");
    }

    //
    //  Beam is absent if not enough signal on the IPM detector
    //
    if (!m_ipm_get_key.empty()) {
      shared_ptr<Psana::Lusi::IpmFexV1> ipm = evt.get(m_ipm_get_key);
      if (ipm.get())
        nobeam |= ipm.get()->sum() < m_ipm_beam_threshold;
    }
  }        

  if (!nobeam) return;

  MsgLog(name(), trace, 
         name() << ": fetching frame data.");

  // example of getting frame data from event
  shared_ptr<Psana::Camera::FrameV1> frame = evt.get(m_get_key);
  if (frame.get()) {
    ndarray<const uint16_t,2> f = frame->data16();
    bool lfatal=false;
    for(unsigned i=0; i<2; i++) {
      if (m_sig_roi_hi[i] >= f.shape()[i]) {
        lfatal |= (m_projectX == (i==0));
        MsgLog(name(), warning, 
               name() << ": signal " << (i==0 ? 'Y':'X') << " upper bound ["
               << m_sig_roi_hi[i] << "] exceeds frame bounds ["
               << f.shape()[i] << "].");
        m_sig_roi_hi[i] = f.shape()[i]-1;
      }
      if (m_sb_roi_hi[i] >= f.shape()[i]) {
        lfatal |= (m_projectX == (i==0));
        MsgLog(name(), warning, 
               name() << ": sideband " << (i==0 ? 'Y':'X') << " upper bound ["
               << m_sb_roi_hi[i] << "] exceeds frame bounds ["
               << f.shape()[i] << "].");
        m_sb_roi_hi[i] = f.shape()[i]-1;
      }
    }
    if (lfatal)
      MsgLog(name(), fatal, 
             name() << ": Fix bounds before proceeding.");
    
    MsgLog(name(), trace, 
           name() << ": projecting.");

    //
    //  Project signal ROI
    //
    unsigned pdim = m_projectX ? 1:0;
    ndarray<const int,1> sig = psalg::project(f, 
                                              m_sig_roi_lo, 
                                              m_sig_roi_hi,
                                              m_pedestal, pdim);

    //
    //  Calculate sideband correction
    //
    ndarray<const int,1> sb;
    if (m_sb_roi_lo[0]!=m_sb_roi_hi[0])
      sb = psalg::project(f, 
                          m_sb_roi_lo , 
                          m_sb_roi_hi,
                          m_pedestal, pdim);

    ndarray<double,1> sigd = make_ndarray<double>(sig.shape()[0]);

    MsgLog(name(), trace, 
           name() << ": correcting for sideband.");

    //
    //  Correct projection for common mode found in sideband
    //
    if (sb.size()) {
      psalg::rolling_average(sb, m_sb_avg, m_sb_avg_fraction);
      
      ndarray<const double,1> sbc = psalg::commonModeLROE(sb, m_sb_avg);

      for(unsigned i=0; i<sig.shape()[0]; i++)
        sigd[i] = double(sig[i])-sbc[i];
    }
    else {
      for(unsigned i=0; i<sig.shape()[0]; i++)
        sigd[i] = double(sig[i]);
    }

    //
    //  Require projection has a minimum amplitude (else no laser)
    //
    bool lcut=true;
    for(unsigned i=0; i<sig.shape()[0]; i++)
      if (sigd[i]>m_proj_cut)
        lcut=false;
      
    if (lcut) return;

    MsgLog(name(), trace, 
           name() << ": Divide by the reference.");

    //
    //  Divide by the reference
    //
    ndarray<double,1> drat = make_ndarray<double>(sig.shape()[0]);
    for(unsigned i=0; i<sig.shape()[0]; i++)
      drat[i] = sigd[i]/m_ref[i] - 1;

    if ((m_count < m_hdump.size()) and m_hmgr) {
      for(unsigned i=0; i<sig.shape()[0]; i++)
        m_hdump[m_count].hraw->fill(double(i)+m_sig_roi_lo[pdim],double(sig[i]));
      for(unsigned i=0; i<sigd.shape()[0]; i++)
        m_hdump[m_count].hcor->fill(double(i)+m_sig_roi_lo[pdim],sigd[i]);
      for(unsigned i=0; i<m_ref.shape()[0]; i++)
        m_hdump[m_count].href->fill(double(i)+m_sig_roi_lo[pdim],m_ref[i]);
      for(unsigned i=0; i<sigd.shape()[0]; i++)
        m_hdump[m_count].hrat->fill(double(i)+m_sig_roi_lo[pdim],drat[i]);
      for(unsigned i=0; i<sigd.shape()[0]; i++) {
        double wt = 1./double(sigd.shape()[0]-i);
        for(unsigned j=0; (i+j)<sigd.shape()[0]; j++)
          m_hdump[m_count].hacf->fill(double(i)+m_sig_roi_lo[pdim],drat[j]*drat[i+j]*wt);
      }
      m_count++;
    }

    MsgLog(name(), trace, 
           name() << ": Accumulate the ACF.");

    //
    //  Accumulate the ACF
    //
    if (m_hmgr) {
      for(unsigned i=0; i<sig.shape()[0]; i++)
        for(unsigned j=i; j<sig.shape()[0]; j++)
          m_hacf->fill(double(j-i),drat[i]*drat[j]);
    }
    psalg::rolling_average(ndarray<const double,1>(sigd),
                           m_ref, m_ref_avg_fraction);
    
  }
  else {
    MsgLog(name(), warning, name() << ": Could not fetch " << m_get_key);
  }
}
 
/// Method which is called at the end of the calibration cycle
void 
Setup::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
Setup::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
Setup::endJob(Event& evt, Env& env)
{
}

} // namespace TimeTool

