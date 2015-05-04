//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisCFD...
//
// Author List:
//      Christopher O'Grady
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/AcqirisCFD.h"
#include "psalg/psalg.h"
#include "psddl_psana/bld.ddl.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ImgAlgos/GlobalMethods.h"
#include "MsgLogger/MsgLogger.h"
#include <iostream>
#include <sstream>
// to work with detector data include corresponding 
// header from psddl_psana package
#include "psddl_psana/acqiris.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(AcqirisCFD)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
AcqirisCFD::AcqirisCFD (const std::string& name)
  : Module(name)
  , m_src()
  , m_str_src()
  , m_key_wform()
  , m_key_wtime()
  , m_key_edges()
  , _evtcount(0)
{
  m_str_src   = configSrc("source", "DetInfo(:Acqiris)");
  m_key_wform = configStr("key_wform", "acqiris_wform");
  m_key_wtime = configStr("key_wtime", "acqiris_wtime");
  m_key_edges = configStr("key_edges", "acqiris_edges_");

  m_baselines = configStr("baselines","");
  m_thresholds = configStr("thresholds","");
  m_fractions = configStr("fractions","");
  m_deadtimes = configStr("deadtimes","");
  m_leading_edges = configStr("leading_edges","");

  if (!m_baselines.empty()) {
    parse_string<double>(m_baselines, v_baseline);
  }
  if (!m_thresholds.empty()) parse_string<double>(m_thresholds, v_threshold);
  if (!m_fractions.empty()) parse_string<double>(m_fractions, v_fraction);
  if (!m_deadtimes.empty()) parse_string<double>(m_deadtimes, v_deadtime);
  if (!m_leading_edges.empty()) parse_string<bool>(m_leading_edges, v_leading_edge);
}

//--------------
// Destructor --
//--------------
AcqirisCFD::~AcqirisCFD ()
{
}

/// Method which is called once at the beginning of the job
void 
AcqirisCFD::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
AcqirisCFD::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
AcqirisCFD::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
AcqirisCFD::event(Event& evt, Env& env)
{
  _evtcount++;
  shared_ptr< ndarray<double,2> > wptr = evt.get(m_str_src, m_key_wform, &m_src);
  const ndarray<double,2> *wf = wptr.get();
  unsigned nchan = wf->shape()[0];
  unsigned nsamples = wf->shape()[1];
  if (wf) {
    shared_ptr< ndarray<double,2> > tptr = evt.get(m_str_src, m_key_wtime, &m_src);
    const ndarray<double,2> &wtime = *tptr;

    for (unsigned i=0; i<nchan; i++) {
      double sampInterval = wtime[i][1]-wtime[i][0];
      ndarray<double,1> onewf = make_ndarray(wf->data()+i*nsamples,nsamples);

      ndarray<double,2> edges;
      edges = psalg::find_edges(onewf,
                                v_baseline[i],
                                v_threshold[i],
                                v_fraction[i],
                                v_deadtime[i]/sampInterval,
                                v_leading_edge[i]);
      unsigned nedges = edges.shape()[0];
      if (nedges) {
        // overwrite the edge "bin" information with "time" information
        for (unsigned j=0; j<nedges; j++) {
          unsigned bin = unsigned(edges[j][0]);
          double binfrac = edges[j][0]-bin;
          double time = wtime[i][bin]+binfrac*sampInterval;
          edges[j][0]= time;
        }
        std::stringstream ss; ss<<i;
        saveNonConst2DArrayInEvent<double> (evt, m_src, m_key_edges+ss.str(), edges);
      }
    }
  }
}
  
/// Method which is called at the end of the calibration cycle
void 
AcqirisCFD::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
AcqirisCFD::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
AcqirisCFD::endJob(Event& evt, Env& env)
{
}

} // namespace ImgAlgos
