//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadArrPeakAnalysis...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadArrPeakAnalysis.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/acqiris.ddl.h"
#include "PSEvt/EventId.h"
#include "PSTime/Time.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace std;
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(CSPadArrPeakAnalysis)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
CSPadArrPeakAnalysis::CSPadArrPeakAnalysis (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
//, m_fname_root()
  , m_print_bits()
  , m_count(0)
  , m_selected(0)
{
  // get the values from configuration or use defaults
  m_str_src    = configSrc("source", "DetInfo(:Cspad)");
  m_key        = configStr("key",   "peaks");
  //m_fname_root = configStr("fname_root","file.root");
  m_print_bits = config   ("print_bits",  0);
}

//--------------
// Destructor --
//--------------
CSPadArrPeakAnalysis::~CSPadArrPeakAnalysis ()
{
}

/// Method which is called once at the beginning of the job
void 
CSPadArrPeakAnalysis::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();

  // initStatistics();
}

/// Method which is called at the beginning of the run
void 
CSPadArrPeakAnalysis::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
CSPadArrPeakAnalysis::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadArrPeakAnalysis::event(Event& evt, Env& env)
{
  ++ m_count;

  if( m_print_bits & 8 ) printEventId(evt);
  procEvent(evt);
}
  
/// Method which is called at the end of the calibration cycle
void 
CSPadArrPeakAnalysis::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadArrPeakAnalysis::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadArrPeakAnalysis::endJob(Event& evt, Env& env)
{
  if( m_print_bits & 2 ) MsgLog(name(), info, "Number of collected in root file events = " << m_selected << " of total " << m_count);
  // summStatistics();
}

//--------------------
//--------------------
//--------------------
//--------------------

void 
CSPadArrPeakAnalysis::procEvent(Event& evt)
{
  shared_ptr< std::vector<Peak> > peaks = evt.get(m_str_src, m_key, &m_src);
  if (peaks.get()) {
    m_peaks = peaks.get();
    if( m_print_bits & 4 ) printPeaks();

    // collectStatistics();
  }
}

//--------------------
/*
void 
CSPadArrPeakAnalysis::initStatistics()
{
  m_tfile = new TFile(m_fname_root.c_str(), "RECREATE", "Created for you by RootManager" );
  m_his01 = new TH1D("hNpeaks","Number of peaks", 100, 0.5, 100.5);
  m_tree  = new TTree("peakTuple", "Tuple for peak data");
  m_branch= m_tree->Branch("peak", &m_peak, "quad/I:sect/I:col/D:row/D:sigma_c/D:sigma_r/D:amax/D:atot/D:btot/D:noise/D:son/D:npix/I");
}
*/
//--------------------
/*
void 
CSPadArrPeakAnalysis::summStatistics()
{
  m_his01 -> Write();
  m_tree  -> Write();
  m_tfile -> Close();
}
*/
//--------------------
/*
void 
CSPadArrPeakAnalysis::collectStatistics()
{
  int npeaks = m_peaks->size();
  m_his01 -> Fill( npeaks );

  for( std::vector<Peak>::const_iterator itv  = m_peaks->begin();
                                         itv != m_peaks->end(); itv++ ) {
    m_peak.quad      = itv->quad;
    m_peak.sect      = itv->sect;    
    m_peak.col       = itv->col;     
    m_peak.row       = itv->row;     
    m_peak.sigma_col = itv->sigma_col;
    m_peak.sigma_row = itv->sigma_row;
    m_peak.ampmax    = itv->ampmax;  
    m_peak.amptot    = itv->amptot;  
    m_peak.bkgdtot   = itv->bkgdtot; 
    m_peak.noise     = itv->noise;  
    m_peak.SoN       = itv->SoN;     
    m_peak.npix      = itv->npix;      

    m_tree  -> Fill();
  }

  ++ m_selected;
}
*/
//--------------------

void 
CSPadArrPeakAnalysis::printPeaks()
{
  MsgLog( name(), info, "Vector of peaks of size " << m_peaks->size() );

  for( std::vector<Peak>::const_iterator itv  = m_peaks->begin();
                                         itv != m_peaks->end(); itv++ ) {
      cout << "  quad     ="   << itv->quad 
           << "  sect     ="   << itv->sect    
           << "  col      ="   << itv->col     
           << "  row      ="   << itv->row     
           << "  sigma_col="   << itv->sigma_col
           << "  sigma_row="   << itv->sigma_row
           << "  ampmax   ="   << itv->ampmax  
           << "  amptot   ="   << itv->amptot  
           << "  bkgdtot  ="   << itv->bkgdtot 
           << "  noise    ="   << itv->noise  
           << "  SoN      ="   << itv->SoN     
           << "  npix     ="   << itv->npix    
           << "\n"; 
  }
}

//--------------------
/// Print input parameters
void 
CSPadArrPeakAnalysis::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource       : "     << m_str_src
	<< "\nkey          : "     << m_key      
      //<< "\nm_fname_root : "     << m_fname_root
        << "\nm_print_bits : "     << m_print_bits;
  }
}

//--------------------

void 
CSPadArrPeakAnalysis::printEventId(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    MsgLog( name(), info, "Event="  << m_count << " ID: " << *eventId);
  }
}

//--------------------

void 
CSPadArrPeakAnalysis::printTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {

    MsgLog( name(), info, " Run="   <<  eventId->run()
                       << " Event=" <<  m_count 
                       << " Time="  <<  eventId->time() );
  }
}

//--------------------


//--------------------

} // namespace ImgAlgos
