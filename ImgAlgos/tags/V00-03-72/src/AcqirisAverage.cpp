//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: AcqirisAverage.cpp 0 2013-10-24 09:00:00Z dubrovin@slac.stanford.edu$
//
// Description:
//	Class AcqirisAverage...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/AcqirisAverage.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <map>
#include <iomanip> // for setw, setfill
#include <sstream> // for stringstream

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
PSANA_MODULE_FACTORY(AcqirisAverage)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
AcqirisAverage::AcqirisAverage (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_ave()
  , m_fname_ave_prefix()  
  , m_thresholds()
  , m_is_positive_signal_list()
  , m_do_inverse_selection_list()
  , m_skip_events()
  , m_proc_events()
  , m_print_bits()
  , m_count_event(0)
  , m_count_get(0)
{
  m_str_src                   = configSrc("source",  "DetInfo(:Acqiris)");
  m_key_in                    = configStr("key_in",          "acq-wform");
  m_key_ave                   = configStr("key_average",       "acq-ave");
  m_fname_ave_prefix          = configStr("fname_ave_prefix",  "acq-ave");
  m_thresholds                = configStr("thresholds",               "");
  m_is_positive_signal_list   = configStr("is_positive_signal",       "");
  m_do_inverse_selection_list = configStr("do_inverse_selection",     "");
  m_skip_events               = config   ("skip_events",               0);
  m_proc_events               = config   ("proc_events",        10000000);
  m_print_bits                = config   ("print_bits",                0);

  m_do_threshold      = (m_thresholds.empty())       ? false : true;
  m_do_save_ave_file  = (m_fname_ave_prefix.empty()) ? false : true;
  m_do_save_ave_evt   = (m_key_ave.empty())          ? false : true;
  m_last_event        = m_skip_events + m_proc_events;
  m_average_is_done   = false;

  if ( m_do_threshold ) {
    parse_string<wform_t>(m_thresholds, v_thresholds);
    parse_string<bool>(m_is_positive_signal_list, v_is_positive_signal);
    parse_string<bool>(m_do_inverse_selection_list, v_do_inverse_selection);
  }

  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

void 
AcqirisAverage::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters    :"
        << "\n source              : " << m_str_src
        << "\n key_in              : " << m_key_in      
        << "\n key_ave             : " << m_key_ave
        << "\n thresholds          : " << m_thresholds
        << "\n is_positive_signal  : " << m_is_positive_signal_list
        << "\n do_inverse_selection: " << m_do_inverse_selection_list
        << "\n skip_events         : " << m_skip_events
        << "\n proc_events         : " << m_proc_events
        << "\n fname_ave_prefix    : " << m_fname_ave_prefix
        << "\n print_bits          : " << m_print_bits
        << "\n do_threshold        : " << m_do_threshold
        << "\n do_save_ave_file    : " << m_do_save_ave_file
        << "\n do_save_ave_evt     : " << m_do_save_ave_evt
        << "\n";     
       }
  printVectorOfThresholds();
}


//--------------
// Destructor --
//--------------
AcqirisAverage::~AcqirisAverage ()
{
}

void 
AcqirisAverage::beginJob(Event& evt, Env& env)
{
}

void 
AcqirisAverage::endJob(Event& evt, Env& env)
{
  if ( ! m_average_is_done ) evaluateAverage(evt); 
}

void 
AcqirisAverage::beginRun(Event& evt, Env& env)
{
  m_str_runnum     = stringRunNumber(evt);
  m_str_experiment = stringExperiment(env);
  m_fname_ave = m_fname_ave_prefix + "-" + m_str_experiment + "-r" + m_str_runnum + "-ave-wfs.txt";
}

void 
AcqirisAverage::beginCalibCycle(Event& evt, Env& env)
{
}

//-------------------------------
// Method which is called with event data
void 
AcqirisAverage::event(Event& evt, Env& env)
{
  m_count_event ++;

  if ( ! isApprovedByCounters() ) return;

  if ( ! procEvent(evt, env) ) return;

  if ( m_count_event == m_last_event ) {
    if( m_print_bits & 16 ) { 
       printVectorOfThresholds();
       printSelectionStatistics();
    }
    evaluateAverage(evt);
  }

  if( m_print_bits & 32 ) {
    std::string msg = (sp_wf) ? 
       stringOf2DArrayData<wform_t>(*sp_wf, std::string("Part of the waveforms:\n"), 0, m_nbrChannels, 10, 20) : "Waveforms is N/A" ; 
    MsgLog( name(), info, msg);
  }
}

//-------------------------------

bool 
AcqirisAverage::isApprovedByCounters()
{
  return (m_count_event < m_skip_events || m_count_event > m_last_event) ? false : true; 
}

//-------------------------------

bool 
AcqirisAverage::procEvent(Event& evt, Env& env)
{
    sp_wf = evt.get(m_str_src, m_key_in, &m_src);

    if (sp_wf.get()) {
        m_count_get ++;
        if (m_count_get == 1) initInFirstEvent(evt, env); 

        ndarray<wform_t,2>& m_wf = *sp_wf;
	//cout << "m_wf.size(): " << m_wf.size() << "\n";

        //------------------------

	if ( m_do_threshold ) {	 

	  //cout << "Apply thresholds";	    

	    for(unsigned c=0; c<m_nbrChannels; c++) {

	        bool threshold_is_crossed = false;

	        wform_t* wf = &m_wf[c][0];
	        wform_t  threshold = v_thresholds[c]; 
                //cout << "  threshold: " << threshold << "\n";

                if (v_is_positive_signal[c]) {
	            for(unsigned s=0; s<m_nbrSamples; s++)
		      if (wf[s] > threshold) { threshold_is_crossed = true; break; }
	        }
		else {
	            for(unsigned s=0; s<m_nbrSamples; s++)
		      if (wf[s] < threshold) { threshold_is_crossed = true; break; }
		}

		if(     (threshold_is_crossed &&  v_do_inverse_selection[c])
		    or (!threshold_is_crossed && !v_do_inverse_selection[c]) ) {

		    // discard waveform - fill it with 0
                    std::fill_n(&m_wf[c][0], int(m_nbrSamples), wform_t(0));
		}
		else {
		    // accumulate statistics
		    m_channel_stat[c] ++;
		    for(unsigned s=0; s<m_nbrSamples; s++) m_wf_sum[c][s] += wf[s];
		}
	    }
	}    

	else { // do not apply threshold

	  //cout << "Do not apply thresholds";	    

	    for(unsigned c=0; c<m_nbrChannels; c++) {
	            wform_t* wf = &m_wf[c][0];
		    m_channel_stat[c] ++;
		    for(unsigned s=0; s<m_nbrSamples; s++) {
                         m_wf_sum[c][s] += wf[s];
			 //if (s<10) cout << " " << wf[s];
		    }
	    }                      //cout << endl; 
	}

        //------------------------

        return true;
    }

    MsgLog(name(), warning, "Object is not found for source: " << m_str_src << " and key: " << m_key_in << "\n" );
    return false;
}

//-------------------------------

void 
AcqirisAverage::initInFirstEvent(Event& evt, Env& env)
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

      m_wf_sum = make_ndarray<wform_t>(m_nbrChannels, m_nbrSamples);
      std::fill_n(m_wf_sum.data(), int(m_size), wform_t(0));    

      m_channel_stat = new unsigned[m_nbrChannels];
      std::fill_n(m_channel_stat, int(m_nbrChannels), unsigned(0));

      if( m_print_bits & 4 ) MsgLog( name(), info, "Begin accumulate statistics since local event = " << m_count_event );  
}

//-------------------------------

void 
AcqirisAverage::evaluateAverage(Event& evt) 
{
  //MsgLog(name(), info, "Begin evaluate average");

  m_wf_ave = make_ndarray<wform_t>(m_nbrChannels, m_nbrSamples);

  for(unsigned c=0; c<m_nbrChannels; c++) {

    if ( m_channel_stat[c] > 0 ) {
      for(unsigned s=0; s<m_nbrSamples; s++)
	m_wf_ave[c][s] = m_wf_sum[c][s] / m_channel_stat[c];
      }

    else std::fill_n(&m_wf_ave[c][0], int(m_nbrSamples), wform_t(0));
  }        

  if (m_do_save_ave_evt) saveNonConst2DArrayInEvent<wform_t> (evt, m_src, m_key_ave, m_wf_ave);
  if (m_do_save_ave_file) save2DArrayInFile<wform_t> (m_fname_ave, m_wf_ave, m_print_bits & 8);

  if( m_print_bits & 4 ) MsgLog( name(), info, "Save average after local event = " << m_count_event );  
  m_average_is_done = true;
}
//-------------------------------

void 
AcqirisAverage::printSelectionStatistics() 
{
    std::map<bool, std::string> msg_add; 
    msg_add[false]="no";
    msg_add[true]="yes";

    stringstream ss;

    ss << "\n  Selection settings:"
       << "\n    is_positive_signal   : " << m_is_positive_signal_list
       << "\n    do_inverse_selection : " << m_do_inverse_selection_list
       << "\n    m_nbrChannels        : " << m_nbrChannels
       << "\n  Statistics of selected waveforms:\n";

    for(unsigned c=0; c<m_nbrChannels; c++) {
       ss << "    Channel: " << setw(2) << c 
          << "  # of selected waveforms: " << setw(6) << m_channel_stat[c];  

       if ( m_do_threshold ) 
            ss << "  threshold: " << v_thresholds[c] << "\n";
       else ss << "  threshold selection is not applied.\n";
    }
    MsgLog(name(), info, ss.str());
}

//-------------------------------

void 
AcqirisAverage::printVectorOfThresholds() 
{
  WithMsgLog(name(), info, msg) {
    msg << "Vector of channel thresholds:";
    for( std::vector<wform_t>::iterator itv  = v_thresholds.begin();
                                        itv != v_thresholds.end(); itv++ ) {      
      msg << "  " << *itv ;
    }
  }
}

//-------------------------------
//-------------------------------
//-------------------------------
} // namespace ImgAlgos
//-------------------------------
//-------------------------------
//-------------------------------
