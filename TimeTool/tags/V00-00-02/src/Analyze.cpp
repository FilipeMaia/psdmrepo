//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeTool::Analyze...
//
// Author List:
//      Matthew J. Weaver
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TimeTool/Analyze.h"

//-----------------
// C/C++ Headers --
//-----------------
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
PSANA_MODULE_FACTORY(Analyze)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace TimeTool {

//----------------
// Constructors --
//----------------
Analyze::Analyze (const std::string& name)
  : Module(name)
{
  // get the values from configuration or use defaults
  m_get_key = configSrc("get_key","DetInfo(:Opal1000)");
  m_put_key = configStr("put_key","TTSPEC");

  m_use_online_config = config("use_online_config",false);

  m_eventcode_nobeam = config("eventcode_nobeam",162);

  m_eventcode_skip   = config("eventcode_skip");

  m_ipm_get_key        = configStr("ipm_get_key");
  m_ipm_beam_threshold = config("ipm_beam_threshold",0.);

  m_calib_poly = configList("calib_poly");

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

  { std::vector<double> w = configList("weights");
    if (w.size()==0) {
      std::string weights_file = configStr("weights_file");
      if (!weights_file.empty()) {
        ifstream s(weights_file.c_str());
        while(s.good()) {
          double v;
          s >> v;
          w.push_back(v);
        }
      }
    }
    if (w.size()==0) {
      MsgLog(name, fatal, name << ": No weights defined for timetool");
    }
    else {
      //  Reverse the ordering of the weights for the
      //  psalg::finite_impulse_response implementation
      m_weights = make_ndarray<double>(w.size());
      for(unsigned i=0; i<w.size(); i++)
        m_weights[i] = w[w.size()-i-1];
    }
  }

  std::string ref_load  = configStr("ref_load");
  if (!ref_load.empty()) {
    ifstream s(ref_load.c_str());
    std::vector<double> ref;
    while(s.good()) {
      double v;
      s >> v;
      ref.push_back(v);
    }
    m_ref = make_ndarray<double>(ref.size());
    for(unsigned i=0; i<ref.size(); i++)
      m_ref[i] = ref[i];
  }

  m_ref_store = configStr("ref_store");

  m_pedestal  = 0;

  m_analyze_event = config("analyze_event",-1);

  unsigned ndump = config("dump",0);
  for(unsigned i=0; i<ndump; i++)
    m_hdump.push_back(DumpH());
}

//--------------
// Destructor --
//--------------
Analyze::~Analyze ()
{
}

/// Method which is called once at the beginning of the job
void 
Analyze::beginJob(Event& evt, Env& env)
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

  unsigned pdim = m_projectX ? 1:0;
  Axis a(m_sig_roi_hi[pdim]-m_sig_roi_lo[pdim]+1,
         double(m_sig_roi_lo[pdim])-0.5,double(m_sig_roi_hi[pdim]+0.5));
  unsigned i=0;
  for(std::list<DumpH>::iterator it=m_hdump.begin();
      it!=m_hdump.end(); it++,i++) {
    { std::stringstream s;
      s << "Raw projection: event " << i;
      it->hraw = env.hmgr().hist1d(s.str().c_str(),"projection",a); }
    { std::stringstream s;
      s << "Ratio: event " << i;
      it->hrat = env.hmgr().hist1d(s.str().c_str(),"ratio",a); }
    { std::stringstream s;
      s << "Filtered: event " << i;
      it->hflt = env.hmgr().hist1d(s.str().c_str(),"filtered",a); }
  }

  m_count=0;
}

/// Method which is called at the beginning of the run
void 
Analyze::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
Analyze::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
Analyze::event(Event& evt, Env& env)
{
  bool nobeam=false;
  { 
    //
    //  Beam is absent if BYKIK fired
    //
    shared_ptr<Psana::EvrData::DataV3> evr = evt.get(Source("DetInfo(:Evr)"));
    if (evr.get()) {
      bool nolaser=false;
      ndarray<const Psana::EvrData::FIFOEvent,1> f = evr.get()->fifoEvents();
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
             name() << ": Failed to retrieve EVR event data.");
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

    if (nobeam) {
      MsgLog(name(), trace, name() << ": Updating reference.");
      psalg::rolling_average(ndarray<const double,1>(sigd),
                             m_ref, m_ref_avg_fraction);

      //
      //  If we are analyzing one event against all references,
      //  copy the cached signal and apply this reference;
      //  else we are done with this event.
      //
      if (!(m_analyze_event<0 || m_count<=m_analyze_event))
        std::copy(m_analyze_signal.begin(),
                  m_analyze_signal.end(),
                  sigd.begin());
      else
        return;
    }

    if (m_ref.size()==0) {
      MsgLog(name(), warning, name() << ": No reference.");
      return;
    }

    if (!(m_analyze_event<0)) {
      //
      //  If this is the selected event to analyze against all 
      //  references, cache it for use from this point on.
      //
      if (m_count==m_analyze_event && !nobeam) {
        m_analyze_signal = make_ndarray<double>(sigd.size());
        std::copy(sigd.begin(), sigd.end(), m_analyze_signal.begin());
      }
      else if (!nobeam) 
        return;
    }

    m_count++;

    //
    //  Divide by the reference
    //
    for(unsigned i=0; i<sig.shape()[0]; i++)
      sigd[i] = sigd[i]/m_ref[i] - 1;

    //
    //  Apply the digital filter
    //
    ndarray<double,1> qwf = psalg::finite_impulse_response(m_weights,sigd);

    if (!m_hdump.empty()) {
      DumpH& h = m_hdump.front();
      for(unsigned i=0; i<sig.shape()[0]; i++)
        h.hraw->fill(double(i),double(sig[i]));
      for(unsigned i=0; i<sigd.shape()[0]; i++)
        h.hrat->fill(double(i),sigd[i]);
      for(unsigned i=0; i<qwf.shape()[0]; i++)
        h.hflt->fill(double(i),qwf[i]);
      m_hdump.pop_front();
    }
    
    //
    //  Find the two highest peaks that are well-separated
    //
    const double afrac = 0.50;
    std::list<unsigned> peaks = 
      psalg::find_peaks(qwf, afrac, 2);

    unsigned nfits = peaks.size();
    if (nfits>0) {
      unsigned ix = *peaks.begin();
      ndarray<double,1> pFit0 = psalg::parab_fit(qwf,ix,0.8);
      if (pFit0[2]>0) {
        double   xflt = pFit0[1]+m_sig_roi_lo[pdim]+m_frame_roi[pdim];
        
        double  xfltc = 0;
        for(unsigned i=m_calib_poly.size(); i!=0; )
          xfltc = xfltc*xflt + m_calib_poly[--i];
        
#define boost_double(v) boost::shared_ptr<double>(boost::make_shared<double>(v))

        evt.put(boost_double(pFit0[0]) ,m_put_key+std::string(":AMPL"));
        evt.put(boost_double(xflt)     ,m_put_key+std::string(":FLTPOS"));
        evt.put(boost_double(xfltc)    ,m_put_key+std::string(":FLTPOS_PS"));
        evt.put(boost_double(pFit0[2]) ,m_put_key+std::string(":FLTPOSFWHM"));
        evt.put(boost_double(m_ref[ix]),m_put_key+std::string(":REFAMPL"));
                
        if (nfits>1) {
          ndarray<double,1> pFit1 = 
            psalg::parab_fit(qwf,*(++peaks.begin()),0.8);
          if (pFit1[2]>0)
            evt.put(boost_double(pFit1[0]), m_put_key+std::string(":AMPLNXT"));
        }

#undef boost_double
      }
    }
  }
  else {
    MsgLog(name(), warning, name() << ": Could not fetch " << m_get_key);
  }
}
 
/// Method which is called at the end of the calibration cycle
void 
Analyze::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
Analyze::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
Analyze::endJob(Event& evt, Env& env)
{
  if (!m_ref_store.empty()) {
    ofstream f(m_ref_store.c_str());
    for(unsigned i=0; i<m_ref.size(); i++)
      f << m_ref[i] << ' ';
    f << std::endl;
  }
}

} // namespace TimeTool

