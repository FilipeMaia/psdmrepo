//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgPeakFilter...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgPeakFilter.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream> // for ofstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
//#include "psddl_psana/acqiris.ddl.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(ImgPeakFilter)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
ImgPeakFilter::ImgPeakFilter (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_sel_mode_str()
  , m_thr_peak()
  , m_thr_total()
  , m_thr_npeaks()
  , m_fname()
  , m_print_bits()
  , m_count(0)
  , m_selected(0)
{
  // get the values from configuration or use defaults
  m_str_src      = configSrc("source",         "DetInfo(:Opal1000)");
  m_key          = configStr("key",            "peaks");
  m_sel_mode_str = configStr("selection_mode", "SELECTION_ON");
  m_thr_peak     = config   ("threshold_peak",  0);
  m_thr_total    = config   ("threshold_total", 0);
  m_thr_npeaks   = config   ("n_peaks_min",     1);
  m_fname        = configStr("fname",          "");
  m_print_bits   = config   ("print_bits",      0);

  setSelectionMode(); // m_sel_mode_str -> enum m_sel_mode 
}

//--------------------

// Print input parameters
void 
ImgPeakFilter::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source     : "     << m_str_src
        << "\n key        : "     << m_key      
        << "\n sel_mode   : "     << m_sel_mode_str  
        << "\n thr_peak   : "     << m_thr_peak
        << "\n thr_total  : "     << m_thr_total
        << "\n thr_npeaks : "     << m_thr_npeaks
        << "\n fname      : "     << m_fname
        << "\n print_bits : "     << m_print_bits
        << "\n";     
  }
}

//--------------------

//--------------
// Destructor --
//--------------
ImgPeakFilter::~ImgPeakFilter ()
{
}

/// Method which is called once at the beginning of the job
void 
ImgPeakFilter::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();  
}

/// Method which is called at the beginning of the run
void 
ImgPeakFilter::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
ImgPeakFilter::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgPeakFilter::event(Event& evt, Env& env)
{
  ++ m_count;

  if ( m_sel_mode == SELECTION_OFF ) { ++ m_selected; return; } // If the filter is OFF then event is selected

  shared_ptr< vector<Peak> > peaks = evt.get(m_str_src, m_key, &m_src);
  if (peaks.get()) {
    m_peaks = peaks.get();
    if ( m_sel_mode == SELECTION_ON  &&  eventIsSelected(evt) ) { doForSelectedEvent(evt); return; } // event is selected
    if ( m_sel_mode == SELECTION_INV && !eventIsSelected(evt) ) { doForSelectedEvent(evt); return; } // event is inversly-selected
  }

  skip(); return; // if event is discarded
}
  
/// Method which is called at the end of the calibration cycle
void 
ImgPeakFilter::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ImgPeakFilter::endRun(Event& evt, Env& env)
{
}

//--------------------
/// Method which is called once at the end of the job
void 
ImgPeakFilter::endJob(Event& evt, Env& env)
{
  if( m_print_bits & 2 ) MsgLog(name(), info, "Job summary: number of selected events = " << m_selected << " of total " << m_count);
}

//--------------------

void 
ImgPeakFilter::setSelectionMode()
{
  m_sel_mode = SELECTION_OFF;
  if (m_sel_mode_str == "SELECTION_ON")  m_sel_mode = SELECTION_ON;
  if (m_sel_mode_str == "SELECTION_INV") m_sel_mode = SELECTION_INV;
}

//--------------------

bool
ImgPeakFilter::eventIsSelected(Event& evt)
{
  if( m_print_bits & 4 ) printPeaks();  
  if( m_print_bits & 16) printEventId(evt);
  return peakSelector();
}

//--------------------

void 
ImgPeakFilter::doForSelectedEvent(Event& evt)
{
  ++ m_selected;
  if( m_print_bits & 8 )           printEventRecord(evt);
  if( m_fname != std::string("") ) savePeaksInFile(evt);
}

//--------------------

//--------------------
// Loop over vector of peaks and count peaks fulfiled the filter conditions
bool
ImgPeakFilter::peakSelector()
{
  m_n_selected_peaks = 0;

  for( vector<Peak>::const_iterator itv  = m_peaks->begin();
                                    itv != m_peaks->end(); itv++ ) {
    if ( itv->ampmax > m_thr_peak 
      && itv->amptot > m_thr_total ) m_n_selected_peaks++;
  }

  if ( m_n_selected_peaks >= m_thr_npeaks ) return true;
  else                                      return false;
}

//--------------------

void 
ImgPeakFilter::printPeaks()
{
  MsgLog( name(), info, "Vector of peaks of size " << m_peaks->size() );

  for( vector<Peak>::iterator itv  = m_peaks->begin();
                              itv != m_peaks->end(); itv++ ) {

      cout << "  x="      << itv->x     
           << "  y="      << itv->y     
           << "  ampmax=" << itv->ampmax
           << "  amptot=" << itv->amptot
           << "  npix="   << itv->npix  
           << endl; 
  }
}

//--------------------

void 
ImgPeakFilter::printEventId(Event& evt)
{
  //shared_ptr<PSEvt::EventId> eventId = evt.get();
  //if (eventId.get()) MsgLog( name(), info, "event ID: " << *eventId );

  MsgLog( name(), info, "r"        << stringRunNumber(evt) 
	             << " "        << stringTimeStamp(evt) 
                     << " evt:"    << stringFromUint(m_count) 
                     << " sel:"    << stringFromUint(m_selected)
	           //<< " npeaks:" << m_n_selected_peaks
  );
}

//--------------------

void 
ImgPeakFilter::printEventRecord(Event& evt)
{
  MsgLog( name(), info, "r"        << stringRunNumber(evt) 
	             << " "        << stringTimeStamp(evt) 
                     << " evt:"    << stringFromUint(m_count) 
                     << " sel:"    << stringFromUint(m_selected)
                     << " npeaks:" << m_n_selected_peaks
  );
}

//--------------------
// Save peak vector info in the file
void 
ImgPeakFilter::savePeaksInFile(Event& evt)
{
  std::string fname; 
  fname = m_fname
        + "-r"    + stringRunNumber(evt) 
        + "-"     + stringTimeStamp(evt) 
    //  + "-ev"   + stringFromUint(m_count)
        + "-peaks.txt";

  MsgLog( name(), info, "Save the peak info in file:" << fname.data() );

  ofstream file; 
  file.open(fname.c_str(),ios_base::out);

  for( vector<Peak>::iterator itv  = m_peaks->begin();
                              itv != m_peaks->end(); itv++ ) {

    //if( m_print_bits & 16 ) printPeakInfo(*itv);

    file << itv->x        << "  "
         << itv->y	  << "  "
         << itv->ampmax	  << "  "
         << itv->amptot	  << "  "
         << itv->npix     << endl; 
  }

  file.close();
}

//--------------------

} // namespace ImgAlgos
