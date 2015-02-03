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
#include "psddl_psana/timetool.ddl.h"
#include "ndarray/ndarray.h"

#include <iomanip>
#include <fstream>
#include <sstream>

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace TimeTool;
PSANA_MODULE_FACTORY(Analyze)

namespace {
  
  // return shared pointer to 1D ndarray of double with one element that is val
  inline boost::shared_ptr<ndarray<double, 1> > boost_double_as_ndarray(double val) {
    unsigned shape[1]={1};
    shared_ptr< ndarray<double,1> > arrPtr = boost::make_shared< ndarray<double,1> >(shape);
    ndarray<double,1> &arr = *arrPtr;
    arr[0]=val;
    return arrPtr;
  }

  static bool 
  calculate_logic(const ndarray<const Psana::TimeTool::EventLogic,1>& cfg,
                  const ndarray<const Psana::EvrData::FIFOEvent,1>& event)
  {
    bool v = (cfg.size()==0) ||
      (cfg[0].logic_op() == Psana::TimeTool::EventLogic::L_AND ||
       cfg[0].logic_op() == Psana::TimeTool::EventLogic::L_AND_NOT);
    for(unsigned i=0; i<cfg.size(); i++) {

      bool p=false;
      for(unsigned j=0; j<event.size(); j++)
        if (event[j].eventCode()==cfg[i].event_code()) {
          p=true;
          break;
        }
      switch(cfg[i].logic_op()) {
      case Psana::TimeTool::EventLogic::L_OR:
        v = v||p; break;
      case Psana::TimeTool::EventLogic::L_AND:
        v = v&&p; break;
      case Psana::TimeTool::EventLogic::L_OR_NOT:
        v = v||!p; break;
      case Psana::TimeTool::EventLogic::L_AND_NOT:
        v = v&&!p; break;
      default: break;
      }
    }
    return v;
  }

  class TimeToolData : public Psana::TimeTool::DataV2 {
  public:
    // class to provide high level TimeTool::DataV2 type to users in the
    // event store. Presently we do not save the three arrays
    // projected_signal, projected_sideband, and projected_reference in this
    // event store data. If all of these are needed, note that the make_shared 
    // pattern used to create an instance of this object has a limit of 9 
    // parameters for the constructor, and those 3 would bring us to 10 parameters.
    TimeToolData(Psana::TimeTool::DataV2::EventType event_type_arg,
                 double amplitude_arg,
                 double position_pixel_arg,
                 double position_time_arg,
                 double position_fwhm_arg,
                 double ref_amplitude_arg,
                 double nxt_amplitude_arg)
      : m_event_type(event_type_arg),
        m_amplitude(amplitude_arg),
        m_position_pixel(position_pixel_arg),
        m_position_time(position_time_arg),
        m_position_fwhm(position_fwhm_arg),
        m_ref_amplitude(ref_amplitude_arg),
        m_nxt_amplitude(nxt_amplitude_arg)
    {}

    virtual enum Psana::TimeTool::DataV2::EventType event_type() const { return m_event_type; }
    virtual double amplitude() const { return m_amplitude; }
    virtual double position_pixel() const { return m_position_pixel; }
    virtual double position_time() const { return m_position_time; };
    virtual double position_fwhm() const { return m_position_fwhm; };
    virtual double ref_amplitude() const { return m_ref_amplitude; };
    virtual double nxt_amplitude() const { return m_nxt_amplitude; };
    virtual ndarray<const int32_t, 1> projected_signal() const { return m_projected_signal; };
    virtual ndarray<const int32_t, 1> projected_sideband() const { return m_projected_sideband; };
    virtual ndarray<const int32_t, 1> projected_reference() const { return m_projected_reference; };

  private:
    Psana::TimeTool::DataV2::EventType m_event_type;
    double m_amplitude;
    double m_position_pixel;
    double m_position_time;
    double m_position_fwhm;
    double m_ref_amplitude;
    double m_nxt_amplitude;
    ndarray<const int32_t, 1> m_projected_signal;
    ndarray<const int32_t, 1> m_projected_sideband;
    ndarray<const int32_t, 1> m_projected_reference;
  };
}


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
  std::string emptys;

  // get the values from configuration or use defaults
  m_get_key = configSrc("get_key","DetInfo(:Opal1000)");
  m_beam_on_off_key = configStr("beam_on_off_key","");
  m_laser_on_off_key = configStr("laser_on_off_key","");
  m_put_key = configStr("put_key","TTSPEC");
  m_put_ndarrays = config("put_ndarrays",false);

  int eventcode_nobeam = config("eventcode_nobeam",0);
  if (eventcode_nobeam>0) {
    ndarray<Psana::TimeTool::EventLogic,1> e = 
      make_ndarray<Psana::TimeTool::EventLogic>(1);
    e[0] = Psana::TimeTool::EventLogic(eventcode_nobeam,
                                       Psana::TimeTool::EventLogic::L_AND_NOT);
    m_beam_logic = e;
  }
  else if (eventcode_nobeam<0) {
    ndarray<Psana::TimeTool::EventLogic,1> e = 
      make_ndarray<Psana::TimeTool::EventLogic>(1);
    e[0] = Psana::TimeTool::EventLogic(-eventcode_nobeam,
                                       Psana::TimeTool::EventLogic::L_AND);
    m_beam_logic = e;
  }

  int eventcode_skip   = config("eventcode_skip",0);
  if (eventcode_skip>0) {
    ndarray<Psana::TimeTool::EventLogic,1> e = 
      make_ndarray<Psana::TimeTool::EventLogic>(1);
    e[0] = Psana::TimeTool::EventLogic(eventcode_skip,
                                       Psana::TimeTool::EventLogic::L_AND_NOT);
    m_laser_logic = e;
  }
  else if (eventcode_skip<0) {
    ndarray<Psana::TimeTool::EventLogic,1> e = 
      make_ndarray<Psana::TimeTool::EventLogic>(1);
    e[0] = Psana::TimeTool::EventLogic(-eventcode_skip,
                                       Psana::TimeTool::EventLogic::L_AND);
    m_laser_logic = e;
  }

  m_ipm_get_key        = configStr("ipm_get_key",emptys);
  m_ipm_beam_threshold = config("ipm_beam_threshold",DBL_MIN);

  { std::vector<double> v = configList("calib_poly");
    if (v.size()) {
      ndarray<double,1> e = make_ndarray<double>(v.size());
      std::copy(v.begin(),v.end(),e.begin());
      m_calib_poly = e;
    } }

  m_projectX_set = !config("projectX",true).isDefault();
  m_projectX     = config("projectX",true);
  m_proj_cut     = config("proj_cut" ,INT_MIN);

  std::vector<unsigned> uempty;
#define initList(l)                                     \
  std::vector<unsigned> l;                              \
  try { l = configList(#l); }                           \
  catch(ConfigSvc::Exception& e) { l = uempty; }

  initList(sig_roi_x)
  initList(sig_roi_y)
  initList(sb_roi_x)
  initList(sb_roi_y)
#undef initList

  m_frame_roi[0] = m_frame_roi[1] = 0;

  if (sb_roi_x.size() && sb_roi_y.size()) {
    m_sb_roi_set = true;
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
    m_sb_roi_set = false;
    m_sb_roi_lo[0] = m_sb_roi_hi[0] = 0;
    m_sb_roi_lo[1] = m_sb_roi_hi[1] = 0;
  }

  if (sig_roi_x.size() && sig_roi_y.size()) {
    m_sig_roi_set   = true;
    m_sig_roi_lo[0] = sig_roi_y[0];
    m_sig_roi_hi[0] = sig_roi_y[1];
    m_sig_roi_lo[1] = sig_roi_x[0];
    m_sig_roi_hi[1] = sig_roi_x[1];
  }
  else {
    m_sig_roi_set   = false;
    m_sig_roi_lo[0] = 0;
    m_sig_roi_hi[0] = 0;
    m_sig_roi_lo[1] = 0;
    m_sig_roi_hi[1] = 0;
  }

  m_ref_roi_set = false;
  m_ref_roi_lo[0] = 0;
  m_ref_roi_hi[0] = 0;
  m_ref_roi_lo[1] = 0;
  m_ref_roi_hi[1] = 0;

  m_sb_avg_fraction  = config("sb_avg_fraction",DBL_MIN);
  m_ref_avg_fraction = config("ref_avg_fraction",DBL_MIN);

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
      //      MsgLog(name, fatal, name << ": No weights defined for timetool");
    }
    else {
      //  Reverse the ordering of the weights for the
      //  psalg::finite_impulse_response implementation
      ndarray<double,1> a = make_ndarray<double>(w.size());
      for(unsigned i=0; i<w.size(); i++)
        a[i] = w[w.size()-i-1];
      m_weights = a;
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
    ref.resize(ref.size()-1);

    m_ref_avg = make_ndarray<double>(ref.size());
    for(unsigned i=0; i<ref.size(); i++)
      m_ref_avg[i] = ref[i];

    MsgLog(name, info, name <<": loaded reference of size " << m_ref_avg.size());
    for(unsigned i=0; i<5; i++) 
      MsgLog(name, info, name <<": ref[" << i << "=" << m_ref_avg[i]);
    MsgLog(name, info, name << ": ..");
    for(unsigned i=m_ref_avg.size()-5; i<m_ref_avg.size(); i++) 
      MsgLog(name, info, name <<": ref[" << i << "=" << m_ref_avg[i]);

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
  m_analyze_projections = false;

  shared_ptr<Psana::TimeTool::ConfigV2> c = 
    env.configStore().get(m_get_key);
  if (c.get()) {
    const Psana::TimeTool::ConfigV2& config = *c.get();
    if (m_beam_logic.size()==0)
      m_beam_logic  = config.beam_logic();
    if (m_laser_logic.size()==0)
      m_laser_logic = config.laser_logic();
    if (m_calib_poly.size()==0)
      m_calib_poly  = config.calib_poly();
    if (!m_projectX_set)
      m_projectX    = unsigned(config.project_axis())==unsigned(Psana::TimeTool::ConfigV1::X);
    if (m_proj_cut != INT_MIN)
      m_proj_cut    = config.signal_cut();
    if (!m_sig_roi_set) {
      m_sig_roi_lo[0] = config.sig_roi_lo().row();
      m_sig_roi_lo[1] = config.sig_roi_lo().column();
      m_sig_roi_hi[0] = config.sig_roi_hi().row();
      m_sig_roi_hi[1] = config.sig_roi_hi().column();
    }
    if (!m_sb_roi_set && config.subtract_sideband()) {
      m_sb_roi_lo[0] = config.sb_roi_lo().row();
      m_sb_roi_lo[1] = config.sb_roi_lo().column();
      m_sb_roi_hi[0] = config.sb_roi_hi().row();
      m_sb_roi_hi[1] = config.sb_roi_hi().column();
    }
    if (m_sb_avg_fraction != DBL_MIN)
      m_sb_avg_fraction = config.sb_convergence();
    if (m_ref_avg_fraction != DBL_MIN)
      m_ref_avg_fraction = config.ref_convergence();
    if (m_weights.size()==0)
      m_weights          = config.weights();

    m_analyze_projections = config.write_projections();

    m_analyze_projections &= 
      m_projectX == (unsigned(config.project_axis())==unsigned(Psana::TimeTool::ConfigV1::X)) &&
      m_sig_roi_lo[0] == config.sig_roi_lo().row() &&
      m_sig_roi_lo[1] == config.sig_roi_lo().column() &&
      m_sig_roi_hi[0] == config.sig_roi_hi().row() &&
      m_sig_roi_hi[1] == config.sig_roi_hi().column();
      
    if (m_sb_roi_set) {
      m_analyze_projections &= 
        config.subtract_sideband() &&
        m_sb_roi_lo[0] == config.sb_roi_lo().row() &&
        m_sb_roi_lo[1] == config.sb_roi_lo().column() &&
        m_sb_roi_hi[0] == config.sb_roi_hi().row() &&
        m_sb_roi_hi[1] == config.sb_roi_hi().column();
    }
  }
  else {
    shared_ptr<Psana::TimeTool::ConfigV1> c = 
      env.configStore().get(m_get_key);
    if (c.get()) {
      const Psana::TimeTool::ConfigV1& config = *c.get();
      if (m_beam_logic.size()==0)
        m_beam_logic  = config.beam_logic();
      if (m_laser_logic.size()==0)
        m_laser_logic = config.laser_logic();
      if (m_calib_poly.size()==0)
        m_calib_poly  = config.calib_poly();
      if (!m_projectX_set)
        m_projectX    = config.project_axis()==Psana::TimeTool::ConfigV1::X;
      if (m_proj_cut != INT_MIN)
        m_proj_cut    = config.signal_cut();
      if (!m_sig_roi_set) {
        m_sig_roi_lo[0] = config.sig_roi_lo().row();
        m_sig_roi_lo[1] = config.sig_roi_lo().column();
        m_sig_roi_hi[0] = config.sig_roi_hi().row();
        m_sig_roi_hi[1] = config.sig_roi_hi().column();
      }
      if (!m_sb_roi_set && config.subtract_sideband()) {
        m_sb_roi_lo[0] = config.sb_roi_lo().row();
        m_sb_roi_lo[1] = config.sb_roi_lo().column();
        m_sb_roi_hi[0] = config.sb_roi_hi().row();
        m_sb_roi_hi[1] = config.sb_roi_hi().column();
      }
      if (m_sb_avg_fraction != DBL_MIN)
        m_sb_avg_fraction = config.sb_convergence();
      if (m_ref_avg_fraction != DBL_MIN)
        m_ref_avg_fraction = config.ref_convergence();
      if (m_weights.size()==0)
        m_weights          = config.weights();

      m_analyze_projections = config.write_projections();

      m_analyze_projections &= 
        m_projectX == (config.project_axis()==Psana::TimeTool::ConfigV1::X) &&
        m_sig_roi_lo[0] == config.sig_roi_lo().row() &&
        m_sig_roi_lo[1] == config.sig_roi_lo().column() &&
        m_sig_roi_hi[0] == config.sig_roi_hi().row() &&
        m_sig_roi_hi[1] == config.sig_roi_hi().column();
      
      if (m_sb_roi_set) {
        m_analyze_projections &= 
          config.subtract_sideband() &&
          m_sb_roi_lo[0] == config.sb_roi_lo().row() &&
          m_sb_roi_lo[1] == config.sb_roi_lo().column() &&
          m_sb_roi_hi[0] == config.sb_roi_hi().row() &&
          m_sb_roi_hi[1] == config.sb_roi_hi().column();
      }
    }
  }

  if (m_beam_logic.size()==0)
    MsgLog(name(), info, 
           name() << ": no beam_logic configuration given.  Assume beam is always T");
  if (m_laser_logic.size()==0)
    MsgLog(name(), info, 
           name() << ": no laser_logic configuration given.  Assume laser is always T");

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
      s << "Ref projection: event " << i;
      it->href = env.hmgr().hist1d(s.str().c_str(),"projection",a); }
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

bool Analyze::getIsOffFromOnOffKey(const std::string & moduleParameter, const std::string & key, Event & evt) 
{
  boost::shared_ptr<std::string> onOff = evt.get(m_get_key, key);
  if (not onOff) {
    onOff = evt.get(key);
    if (not onOff) MsgLog(name(), fatal, moduleParameter << "=" << key
                          << " specified but not present for either the source specified with the get_key="
                          << m_get_key << " nor no source. It must be present in all events ");
  }
  bool isOn = *onOff == "on";
  bool isOff = *onOff == "off";
  if ((not isOn) and (not isOff)) MsgLog(name(), fatal, key
                                         << " found but value must be on of 'on' or 'off', "
                                         << "all lowercase. However it is: " << *onOff);
  return isOff;
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
Analyze::event(Event& evt, Env& env)
{
  shared_ptr<Psana::EvrData::DataV3> evr = evt.get(Source("DetInfo(:Evr)"));
  if (!evr.get()) {
    MsgLog(name(), warning, name() << ": Could not fetch evr data");
    return;
  }

  bool nobeam;
  bool nolaser;
  
  if (m_beam_on_off_key.length()>0) {
    nobeam = getIsOffFromOnOffKey("beam_on_off_key", 
                                  m_beam_on_off_key, evt);
  } else {
    nobeam  = !calculate_logic(m_beam_logic,
                                  evr.get()->fifoEvents());
  }
  

  if (m_laser_on_off_key.length()>0) {
    nolaser = getIsOffFromOnOffKey("laser_on_off_key",
                                   m_laser_on_off_key, evt);
  } else {
    nolaser = !calculate_logic(m_laser_logic, 
                               evr.get()->fifoEvents());
  }

  bool use_sb_roi  = (m_sb_roi_lo [0]!=m_sb_roi_hi [0]);
  bool use_ref_roi = (m_ref_roi_lo[0]!=m_ref_roi_hi[0]);

  MsgLog(name(), trace, name() << ": evr_codes " << evr.get()->fifoEvents().size()
         << ": nobeam " << (nobeam ? 'T':'F')
         << ": nolaser " << (nolaser ? 'T':'F'));

  if (nolaser) return;

  //
  //  Beam is absent if not enough signal on the IPM detector
  //
  if (!m_ipm_get_key.empty()) {
    shared_ptr<Psana::Lusi::IpmFexV1> ipm = evt.get(Source(m_ipm_get_key));
    if (ipm.get()) {
      MsgLog(name(), info, name() << ": ipm sum = " << ipm.get()->sum());
      nobeam |= ipm.get()->sum() < m_ipm_beam_threshold;
    }
    else {
      MsgLog(name(), info, name() << ": failed to get ipm " << m_ipm_get_key);

      MsgLog(name(), info, name() << ":\t  src   \tkey\talias");
      std::list<EventKey> keys = evt.keys();
      for(std::list<EventKey>::iterator it=keys.begin(); it!=keys.end(); it++) {
        MsgLog(name(), info, name() << ": "
               << '\t' << std::setw(8) << std::hex << it->src().phy() 
               << '\t' << it->key()
               << '\t' << it->alias());
      }
    }
  }

  ndarray<const int32_t,1> sig;
  ndarray<const int32_t,1> sb;
  ndarray<const int32_t,1> ref;
  unsigned pdim = m_projectX ? 1:0;

  if (m_analyze_projections) {
    shared_ptr<Psana::TimeTool::DataV2> tt = evt.get(m_get_key);
    if (tt.get()) {
      sig  = tt.get()->projected_signal();
      sb   = tt.get()->projected_sideband();
      ref  = tt.get()->projected_reference();
    }
    else {
      shared_ptr<Psana::TimeTool::DataV1> tt = evt.get(m_get_key);
      if (tt.get()) {
        sig  = tt.get()->projected_signal();
        sb   = tt.get()->projected_sideband();
      }
      else {
        MsgLog(name(), warning, name() << ": Could not fetch timetool data");
        return;
      }
    }
  }
  else {
    // example of getting frame data from event
    shared_ptr<Psana::Camera::FrameV1> frame = evt.get(m_get_key);
    if (!frame.get()) {
      MsgLog(name(), warning, name() << ": Could not fetch frame data");
      return;
    }

    m_pedestal = frame->offset();

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
      if (m_ref_roi_hi[i] >= f.shape()[i]) {
        lfatal |= (m_projectX == (i==0));
        MsgLog(name(), warning, 
               name() << ": reference " << (i==0 ? 'Y':'X') << " upper bound ["
               << m_ref_roi_hi[i] << "] exceeds frame bounds ["
               << f.shape()[i] << "].");
        m_ref_roi_hi[i] = f.shape()[i]-1;
      }
    }
    if (lfatal)
      MsgLog(name(), fatal, 
             name() << ": Fix bounds before proceeding.");
    
    //
    //  Project signal ROI
    //
    sig = psalg::project(f, 
                         m_sig_roi_lo, 
                         m_sig_roi_hi,
                         m_pedestal, pdim);
    
    //
    //  Calculate sideband correction
    //
    if (use_sb_roi)
      sb = psalg::project(f, 
                          m_sb_roi_lo , 
                          m_sb_roi_hi,
                          m_pedestal, pdim);

    //
    //  Calculate reference correction
    //
    if (use_ref_roi)
      ref = psalg::project(f, 
                           m_ref_roi_lo , 
                           m_ref_roi_hi,
                           m_pedestal, pdim);
  }
    
  ndarray<double,1> sigd = make_ndarray<double>(sig.shape()[0]);
  ndarray<double,1> refd = make_ndarray<double>(sig.shape()[0]);

  //
  //  Correct projection for common mode found in sideband
  //
  if (sb.size()) {
    psalg::rolling_average(sb, m_sb_avg, m_sb_avg_fraction);
      
    ndarray<const double,1> sbc = psalg::commonModeLROE(sb, m_sb_avg);

    if (use_ref_roi)
      for(unsigned i=0; i<sig.shape()[0]; i++) {
	sigd[i] = double(sig[i])-sbc[i];
	refd[i] = double(ref[i])-sbc[i];
      }
    else
      for(unsigned i=0; i<sig.shape()[0]; i++)
        sigd[i] = double(sig[i])-sbc[i];
  }
  else {
    if (use_ref_roi)
      for(unsigned i=0; i<sig.shape()[0]; i++) {
	sigd[i] = double(sig[i]);
	refd[i] = double(ref[i]);
      }
    else
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
    psalg::rolling_average(ndarray<const double,1>(use_ref_roi? refd:sigd),
                           m_ref_avg, m_ref_avg_fraction);
    
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
  else if (use_ref_roi) {
    psalg::rolling_average(ndarray<const double,1>(refd),
                           m_ref_avg, m_ref_avg_fraction);
  }

  if (m_ref_avg.size()==0) {
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
    sigd[i] = sigd[i]/m_ref_avg[i] - 1;

  //
  //  Apply the digital filter
  //
  ndarray<double,1> qwf = psalg::finite_impulse_response(m_weights,sigd);

  if (!m_hdump.empty()) {
    DumpH& h = m_hdump.front();
    for(unsigned i=0; i<sig.shape()[0]; i++)
      h.hraw->fill(double(i)+m_sig_roi_lo[pdim],double(sig[i]));
    for(unsigned i=0; i<sig.shape()[0]; i++)
      h.href->fill(double(i)+m_sig_roi_lo[pdim],m_ref_avg[i]);
    for(unsigned i=0; i<sigd.shape()[0]; i++)
      h.hrat->fill(double(i)+m_sig_roi_lo[pdim],sigd[i]);
    for(unsigned i=0; i<qwf.shape()[0]; i++)
      h.hflt->fill(double(i)+m_sig_roi_lo[pdim],qwf[i]);
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

      if (not m_put_ndarrays) {
        evt.put(boost_double(pFit0[0]) ,m_put_key+std::string(":AMPL"));
        evt.put(boost_double(xflt)     ,m_put_key+std::string(":FLTPOS"));
        evt.put(boost_double(xfltc)    ,m_put_key+std::string(":FLTPOS_PS"));
        evt.put(boost_double(pFit0[2]) ,m_put_key+std::string(":FLTPOSFWHM"));
        evt.put(boost_double(m_ref_avg[ix]),m_put_key+std::string(":REFAMPL"));
      } else {
        evt.put(boost_double_as_ndarray(pFit0[0]) ,m_put_key+std::string(":AMPL"));
        evt.put(boost_double_as_ndarray(xflt)     ,m_put_key+std::string(":FLTPOS"));
        evt.put(boost_double_as_ndarray(xfltc)    ,m_put_key+std::string(":FLTPOS_PS"));
        evt.put(boost_double_as_ndarray(pFit0[2]) ,m_put_key+std::string(":FLTPOSFWHM"));
        evt.put(boost_double_as_ndarray(m_ref_avg[ix]),m_put_key+std::string(":REFAMPL"));
      }
                
      double nxtAmpl=0;
      if (nfits>1) {
        ndarray<double,1> pFit1 = 
          psalg::parab_fit(qwf,*(++peaks.begin()),0.8);
        if (pFit1[2]>0)
          nxtAmpl = pFit1[0];
          if (not m_put_ndarrays) {
            evt.put(boost_double(nxtAmpl), m_put_key+std::string(":AMPLNXT"));
          } else {
            evt.put(boost_double_as_ndarray(nxtAmpl), m_put_key+std::string(":AMPLNXT"));
          }
      }

      // put 
      boost::shared_ptr<Psana::TimeTool::DataV2> timeToolEvtData = 
        boost::make_shared<TimeToolData>(Psana::TimeTool::DataV2::Signal,
                                         pFit0[0],
                                         xflt,
                                         xfltc,
                                         pFit0[2],
                                         m_ref_avg[ix],
                                         nxtAmpl);
      evt.put(timeToolEvtData,m_put_key);
      
#undef boost_double
    }
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
    for(unsigned i=0; i<m_ref_avg.size(); i++)
      f << m_ref_avg[i] << ' ';
    f << std::endl;
  }
}

} // namespace TimeTool

