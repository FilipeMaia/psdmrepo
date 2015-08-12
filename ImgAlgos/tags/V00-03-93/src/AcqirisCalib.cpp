//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: AcqirisCalib.cpp 0 2013-10-24 09:00:00Z dubrovin@slac.stanford.edu$
//
// Description:
//	Class AcqirisCalib...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/AcqirisCalib.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <map>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std;
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(AcqirisCalib)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
AcqirisCalib::AcqirisCalib (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out()
  , m_fname_bline()  
  , m_skip_events()
  , m_proc_events()
  , m_print_bits()
  , m_count_event(0)
  , m_count_get(0)
{
  m_str_src           = configSrc("source",      "DetInfo(:Acqiris)");
  m_key_in            = configStr("key_in",              "acq_wform");
  m_key_out           = configStr("key_out",         "wf-calibrated");
  m_fname_bline       = configStr("fname_base_line",       "acq-ave");
  m_skip_events       = config   ("skip_events",                   0);
  m_proc_events       = config   ("proc_events",            10000000);
  m_print_bits        = config   ("print_bits",                    0);

  m_do_subtr_baseline = (m_fname_bline.empty())    ? false : true;
  m_last_event        = m_skip_events + m_proc_events;

  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

void 
AcqirisCalib::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters    :"
        << "\n source              : " << m_str_src
        << "\n key_in              : " << m_key_in      
        << "\n key_out             : " << m_key_out
        << "\n skip_events         : " << m_skip_events
        << "\n proc_events         : " << m_proc_events
        << "\n fname_bline         : " << m_fname_bline
        << "\n do_subtr_baseline   : " << m_do_subtr_baseline
        << "\n";     
       }
}

//--------------
// Destructor --
//--------------
AcqirisCalib::~AcqirisCalib ()
{
}

void 
AcqirisCalib::beginJob(Event& evt, Env& env)
{
}

void 
AcqirisCalib::endJob(Event& evt, Env& env)
{
}

void 
AcqirisCalib::beginRun(Event& evt, Env& env)
{
  m_str_runnum     = stringRunNumber(evt);
  m_str_experiment = stringExperiment(env);
  m_fname_bline_ex = m_fname_bline + "-" + m_str_experiment + "-r" + m_str_runnum + "-ave-wfs.txt";
}

void 
AcqirisCalib::beginCalibCycle(Event& evt, Env& env)
{
}

//-------------------------------
// Method which is called with event data
void 
AcqirisCalib::event(Event& evt, Env& env)
{
  m_count_event ++;

  if ( ! isApprovedByCounters() ) return;

  if ( ! procEvent(evt, env) ) return;

  if ( m_count_event == m_last_event and m_print_bits & 4 ) MsgLog( name(), info, "Do not subtract baseline since local event = " << m_count_event );  

  if( m_print_bits & 16 ) printPartOfInputWaveform2DArray();
}

//-------------------------------

bool 
AcqirisCalib::isApprovedByCounters()
{
  return (m_count_event < m_skip_events || m_count_event > m_last_event) ? false : true; 
}

//-------------------------------

bool 
AcqirisCalib::procEvent(Event& evt, Env& env)
{
    sp_wf = evt.get(m_str_src, m_key_in, &m_src);

    if (sp_wf.get()) {
        m_count_get ++;
        if (m_count_get==1) initInFirstEvent(evt, env); 


        m_wf = make_ndarray<wform_t>(m_nbrChannels, m_nbrSamples);

        ndarray<wform_t,2>& m_wf_data = *sp_wf;

        //cout << "m_wf.size(): " << m_wf.size() << "\n";

        //------------------------

	if ( m_do_subtr_baseline ) {	 

	  //cout << "    Subtract baseline...\n"; 

	  for(unsigned c=0; c<m_nbrChannels; c++)
	    for(unsigned s=0; s<m_nbrSamples; s++)
	      m_wf[c][s] = m_wf_data[c][s] - m_wf_bline[c][s];

          saveNonConst2DArrayInEvent<wform_t> (evt, m_src, m_key_out, m_wf);
	}

	else 
          saveNonConst2DArrayInEvent<wform_t> (evt, m_src, m_key_out, m_wf_data);

        //------------------------

        return true;
    }

    MsgLog(name(), warning, "Object is not found for source: " << m_str_src << " and key: " << m_key_in << "\n" );
    return false;
}

//-------------------------------

void 
AcqirisCalib::initInFirstEvent(Event& evt, Env& env)
{
      m_nbrChannels = sp_wf->shape()[0];
      m_nbrSamples  = sp_wf->shape()[1];
      m_size        = sp_wf->size();

      if( m_print_bits & 2 )
          MsgLog( name(), info,
                  "shape = "     << m_nbrChannels 
                  << ", "        << m_nbrSamples 
                  << ", size = " << m_size 
		 )

      m_wf_bline = make_ndarray<wform_t>(m_nbrChannels, m_nbrSamples);
      if ( m_do_subtr_baseline ) loadBaseLine2DArrayFromFile();
      else std::fill_n(m_wf_bline.data(), int(m_size), wform_t(0));
      if( m_print_bits & 32 ) printPartOfBaseLine2DArray();
      if( m_print_bits &  4 ) MsgLog( name(), info, "Begin subtract baseline since local event = " << m_count_event );  
}


//-------------------------------

void 
AcqirisCalib::loadBaseLine2DArrayFromFile()
{
  std::map <bool, std::string> msg_ext; msg_ext[false]=" does not exist.";  msg_ext[true]= " exists.";
  if ( m_print_bits & 8 ) MsgLog( name(), info, "Expected base-line file: " << m_fname_bline    << msg_ext[file_exists(m_fname_bline)] );
  if ( m_print_bits & 8 ) MsgLog( name(), info, "Expected base-line file: " << m_fname_bline_ex << msg_ext[file_exists(m_fname_bline_ex)] );

  std::string fname_bline = "";
  if      ( file_exists(m_fname_bline_ex) ) fname_bline = m_fname_bline_ex;
  else if ( file_exists(m_fname_bline) )    fname_bline = m_fname_bline;
  else { MsgLog( name(), warning, "Requested base line file " << m_fname_bline 
                                  << " or its extension "     << m_fname_bline_ex 
                                  << "do not exist... Will use 0 for baseline subtraction...");
         return;
  }

  load2DArrayFromFile<wform_t>(fname_bline, m_wf_bline, m_print_bits & 8);
}

//-------------------------------

void 
AcqirisCalib::printPartOfInputWaveform2DArray()
{
  std::string msg = (sp_wf) ? stringOf2DArrayData<wform_t>(*sp_wf.get(), std::string("Part of the waveforms:\n"), 0, m_nbrChannels, 0, 10) : "Waveforms is N/A"; 
  MsgLog( name(), info, msg);
}

//-------------------------------

void 
AcqirisCalib::printPartOfBaseLine2DArray()
{
  std::string msg = stringOf2DArrayData<wform_t>(m_wf_bline, std::string("Part of the base line array:\n"), 0, m_nbrChannels, 0, 10);
  MsgLog( name(), info, msg );
}

//-------------------------------
} // namespace ImgAlgos
//-------------------------------
//-------------------------------
//-------------------------------
