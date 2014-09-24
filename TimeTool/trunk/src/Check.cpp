//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeTool::Check...
//
// Author List:
//      Matthew J. Weaver
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TimeTool/Check.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
// to work with detector data include corresponding 
// header from psddl_psana package
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
#include "psddl_psana/epics.ddl.h"
#include "psddl_psana/bld.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace TimeTool;
PSANA_MODULE_FACTORY(Check)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace TimeTool {

//----------------
// Constructors --
//----------------
Check::Check (const std::string& name)
  : Module(name)
{
  // get the values from configuration or use defaults
  m_get_key1 = configStr("get_key1");
  m_get_key2 = configStr("get_key2");

  m_angle_shift        = configStr("angle_shift","LAS:FS2:REG:Angle:Shift:rd");
  m_angle_shift_offset = config("angle_shift_offset",1.27e7);
  m_phcav1_limits = configList("phcav1_limits");
  m_phcav2_limits = configList("phcav2_limits");
  m_tt_calib      = configList("tt_calib");

  m_amplitude_binning = configList("amplitude_binning");
  m_position_binning  = configList("position_binning");
  m_width_binning     = configList("width_binning");

  if (m_phcav1_limits.size()==0) {
    m_phcav1_limits.push_back(-2);
    m_phcav1_limits.push_back( 2);
  }

  if (m_phcav2_limits.size()==0) {
    m_phcav2_limits.push_back(-2);
    m_phcav2_limits.push_back( 2);
  }

  if (m_tt_calib.size()==0) {
    m_tt_calib.push_back(0);
    m_tt_calib.push_back(1);
  }
}

//--------------
// Destructor --
//--------------
Check::~Check ()
{
}

/// Method which is called once at the beginning of the job
void 
Check::beginJob(Event& evt, Env& env)
{
#define axis_from_vector(v) Axis(unsigned(v[0]),v[1],v[2])

  m_ampl = env.hmgr().hist1d("Amplitude","ampl" ,
                             m_amplitude_binning.size() ?
                             axis_from_vector(m_amplitude_binning) :
                             Axis(100,-0.02,0.18));
  m_fltp = env.hmgr().hist1d("Position" ,"pos"  ,
                             m_position_binning.size() ?
                             axis_from_vector(m_position_binning) :
                             Axis(100,0.,1000));
  m_fltw = env.hmgr().hist1d("Width"    ,"width",
                             m_width_binning.size() ?
                             axis_from_vector(m_width_binning) :
                             Axis(100,0.,100));
  m_ampl_v_fltp = env.hmgr().hist2d("Amplitude v Position",
                                    "amplvpos",
                                    Axis(100,0.,1000),
                                    Axis(100,-0.01,0.09));
  m_namp = env.hmgr().hist1d("Next Ampl","ampl" ,Axis(100,-0.01,0.09));
  m_namp2 = env.hmgr().hist2d("Next Amplitude v Amplitude",
                              "nampvampl",
                              Axis(100,-0.02,0.18),
                              Axis(100,-0.01,0.09));

  Axis a(800,-4.,4.);
  m_p1_v_p2  = env.hmgr().prof1("phcav2 v phcav1","phase cavity corr",a);
  m_pos_v_p1 = env.hmgr().prof1("pos v phcav1","timetool pos1",a);
  m_pos_v_p2 = env.hmgr().prof1("pos v phcav2","timetool pos2",a);
  m_tt_v_p1  = env.hmgr().prof1("tt v phcav1","timetool corr1",a);
  m_tt_v_p2  = env.hmgr().prof1("tt v phcav2","timetool corr2",a);
  m_p1_m_p2  = env.hmgr().hist1d("phcav2 - phcav1","phase cavity diff",a);
  m_tt_m_p1  = env.hmgr().hist1d("tt - phcav1","timetool diff1",a);
  m_tt_m_p2  = env.hmgr().hist1d("tt - phcav2","timetool diff2",a);
  m_tt_v_p2_2d  = env.hmgr().hist2i("tt v phcav2 2d","timetool corr2",
                                    Axis(60,-0.3,0.3), Axis(80,640.,800.));
}

/// Method which is called at the beginning of the run
void 
Check::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
Check::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
Check::event(Event& evt, Env& env)
{
  double ampl1, fltp1, fltw1, ramp1, namp1;

  shared_ptr<double> q = evt.get(m_get_key1+std::string(":AMPL"));
  if (q.get()) {
    ampl1 = *q;
    fltp1 = *(q = evt.get(m_get_key1+std::string(":FLTPOS")));
    fltw1 = *(q = evt.get(m_get_key1+std::string(":FLTPOSFWHM")));
    ramp1 = *(q = evt.get(m_get_key1+std::string(":REFAMPL")));
    namp1 = (q = evt.get(m_get_key1+std::string(":AMPLNXT"))) ? *q : 0;
  }
  else {
    shared_ptr< ndarray<double,1> > p = 
      evt.get(m_get_key1+std::string(":AMPL"));
    if (p.get()) {
      ampl1 = *p->data();
      fltp1 = *(p = evt.get(m_get_key1+std::string(":FLTPOS")))->data();
      fltw1 = *(p = evt.get(m_get_key1+std::string(":FLTPOSFWHM")))->data();
      ramp1 = *(p = evt.get(m_get_key1+std::string(":REFAMPL")))->data();
      namp1 = (p = evt.get(m_get_key1+std::string(":AMPLNXT"))) ? *p->data() : 0;
    }
    else
      return;
  }


  double angsh = 0;

  EpicsStore& epics = env.epicsStore();
  try {
    angsh = epics.value(m_angle_shift);
    angsh -= m_angle_shift_offset;
    angsh *= 1.e-3;
  } catch(PSEnv::Exception& e) {
  }

  m_ampl->fill(ampl1);
  m_fltp->fill(fltp1);
  m_fltw->fill(fltw1);
  m_ampl_v_fltp->fill(fltp1,ampl1);
  
  m_namp->fill(namp1);
  m_namp2->fill(ampl1,namp1);

  boost::shared_ptr<Psana::Bld::BldDataPhaseCavity> phcav = 
    evt.get(Source("BldInfo(PhaseCavity)"));
  if (phcav.get()) {
    double  tt = 0;
    for(unsigned i=m_tt_calib.size(); i!=0; )
      tt = tt*fltp1 + m_tt_calib[--i];

    bool lphcav1 = (phcav->fitTime1()>m_phcav1_limits[0] &&
                    phcav->fitTime1()<m_phcav1_limits[1]);
    bool lphcav2 = (phcav->fitTime2()>m_phcav2_limits[0] &&
                    phcav->fitTime2()<m_phcav2_limits[1]);

    if (lphcav1 && lphcav2) {
      m_p1_v_p2->fill(phcav->fitTime1(),phcav->fitTime2());
      m_p1_m_p2->fill(phcav->fitTime2()-phcav->fitTime1());
    }

    if (lphcav1) {
      m_pos_v_p1->fill(-phcav->fitTime1(),fltp1);
      m_tt_v_p1->fill(angsh-phcav->fitTime1(),tt);
      m_tt_m_p1->fill(tt-angsh+phcav->fitTime1());
    }

    if (lphcav2) {
      m_pos_v_p2->fill(-phcav->fitTime2(),fltp1);
      m_tt_v_p2->fill(angsh-phcav->fitTime2(),tt);
      m_tt_m_p2->fill(tt-angsh+phcav->fitTime2());
      m_tt_v_p2_2d->fill(angsh-phcav->fitTime2(),fltp1);
    }
  }

  if (m_get_key2.empty()) return;

  try {
    double ampl2 = epics.value(m_get_key2+std::string(":AMPL"));
    double fltp2 = epics.value(m_get_key2+std::string(":FLTPOS"));
    double fltw2 = epics.value(m_get_key2+std::string(":FLTPOSFWHM"));
    double ramp2 = epics.value(m_get_key2+std::string(":REFAMPL"));
    double namp2 = epics.value(m_get_key2+std::string(":AMPLNXT"));
    
#define DEREF(p) p

    MsgLog(name(), info, 
           name() << ": ttparms " << std::endl
           << "\t Ampl " << DEREF(ampl1) << '/' << ampl2 << std::endl
           << "\t Pos  " << DEREF(fltp1) << '/' << fltp2 << std::endl
           << "\t FWHM " << DEREF(fltw1) << '/' << fltw2 << std::endl
           << "\t RAmp " << DEREF(ramp1) << '/' << ramp2 << std::endl
           << "\t NAmp " << DEREF(namp1) << '/' << namp2 << std::endl);

#undef DEREF
  } catch(PSEnv::Exception& e) {
    MsgLog(name(), info, 
           name() << ": unable to get epics values for " << m_get_key2);
  }
}
 
/// Method which is called at the end of the calibration cycle
void 
Check::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
Check::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
Check::endJob(Event& evt, Env& env)
{
}

} // namespace TimeTool
