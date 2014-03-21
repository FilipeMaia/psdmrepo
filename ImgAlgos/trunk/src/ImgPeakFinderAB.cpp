//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ImgPeakFinderAB...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/ImgPeakFinderAB.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <cmath>
#include <iomanip> // for setw, setfill
#include <sstream> // for stringstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "ImgAlgos/GlobalMethods.h"

#include "PSEvt/EventId.h"
#include "cspad_mod/DataT.h"
#include "cspad_mod/ElementT.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(ImgPeakFinderAB)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

ImgPeakFinderAB::ImgPeakFinderAB (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_key_signal_out()  
  , m_key_peaks_out()  
  , m_maskFile_inp()
  , m_maskFile_out()
  , m_fracFile_out()
  , m_evtFile_out()
  , m_rmin()
  , m_dr()
  , m_SoNThr_noise()
  , m_SoNThr_signal()
  , m_frac_noisy_imgs()
  , m_peak_npix_min()
  , m_peak_npix_max()
  , m_peak_amp_tot_thr()
  , m_peak_SoN_thr()
  , m_event_npeak_min()
  , m_event_npeak_max()
  , m_event_amp_tot_thr()
  , m_nevents_mask_update()
  , m_nevents_mask_accum()
  , m_sel_mode_str()
  , m_out_file_bits()
  , m_print_bits()
  , m_count(0)
  , m_count_selected(0)
  , m_count_mask_update(0)
  , m_count_mask_accum(0)
{
  // get the values from configuration or use defaults
  m_str_src           = configSrc("source",     "DetInfo(:Cspad)");
  m_key               = configStr("key",        "");                 //"calibrated"
  m_key_signal_out    = configStr("key_signal_out", "");
  m_key_peaks_out     = configStr("key_peaks_out", "peaks");
  m_maskFile_inp      = configStr("hot_pix_mask_inp_file", ""); // "cspad-pix-mask-in.dat"
  m_maskFile_out      = configStr("hot_pix_mask_out_file", "cspad-pix-mask-out.dat");
  m_fracFile_out      = configStr("frac_noisy_evts_file",  "cspad-pix-frac-out.dat");
  m_evtFile_out       = configStr("evt_file_out",          "./cspad-ev-");
  m_rmin              = config   ("rmin",                    3 );
  m_dr                = config   ("dr",                      1 );
  m_SoNThr_noise      = config   ("SoNThr_noise",            3 );
  m_SoNThr_signal     = config   ("SoNThr_signal",           5 );
  m_frac_noisy_imgs   = config   ("frac_noisy_imgs",       0.9 ); 

  m_peak_npix_min     = config   ("peak_npix_min",           4 );
  m_peak_npix_max     = config   ("peak_npix_max",          25 );
  m_peak_amp_tot_thr  = config   ("peak_amp_tot_thr",        0.);
  m_peak_SoN_thr      = config   ("peak_SoN_thr",            7.);

  m_event_npeak_min   = config   ("event_npeak_min",        10 );
  m_event_npeak_max   = config   ("event_npeak_max",     10000 );
  m_event_amp_tot_thr = config   ("event_amp_tot_thr",       0.);

  m_nevents_mask_update= config   ("nevents_mask_update",    0 );
  m_nevents_mask_accum = config   ("nevents_mask_accum",    50 );

  m_sel_mode_str      = configStr("selection_mode", "SELECTION_ON");
  m_out_file_bits     = config   ("out_file_bits",           0 ); 
  m_print_bits        = config   ("print_bits",              0 );

  // initialize arrays
  setSelectionMode(); // m_sel_mode_str -> enum m_sel_mode 

}

//--------------
// Destructor --
//--------------
ImgPeakFinderAB::~ImgPeakFinderAB ()
{
}
//--------------------

// Print input parameters
void 
ImgPeakFinderAB::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source                : " << m_str_src
        << "\n key                   : " << m_key      
        << "\n m_key_signal_out      : " << m_key_signal_out
        << "\n m_key_peaks_out       : " << m_key_peaks_out 
        << "\n m_maskFile_inp        : " << m_maskFile_inp    
        << "\n m_maskFile_out        : " << m_maskFile_out    
        << "\n m_fracFile_out        : " << m_fracFile_out    
        << "\n m_evtFile_out         : " << m_evtFile_out  
        << "\n m_rmin                : " << m_rmin    
        << "\n m_dr                  : " << m_dr     
        << "\n m_SoNThr_noise        : " << m_SoNThr_noise     
        << "\n m_SoNThr_signal       : " << m_SoNThr_signal     
        << "\n m_frac_noisy_imgs     : " << m_frac_noisy_imgs    
        << "\n m_peak_npix_min       : " << m_peak_npix_min     
        << "\n m_peak_npix_max       : " << m_peak_npix_max     
        << "\n m_peak_amp_tot_thr    : " << m_peak_amp_tot_thr  
        << "\n m_peak_SoN_thr        : " << m_peak_SoN_thr  
        << "\n m_event_npeak_min     : " << m_event_npeak_min   
        << "\n m_event_npeak_max     : " << m_event_npeak_max   
        << "\n m_event_amp_tot_thr   : " << m_event_amp_tot_thr 
        << "\n m_nevents_mask_update : " << m_nevents_mask_update
        << "\n m_nevents_mask_accum  : " << m_nevents_mask_accum 
        << "\n m_sel_mode_str        : " << m_sel_mode_str 
        << "\n m_sel_mode            : " << m_sel_mode 
        << "\n m_out_file_bits       : " << m_out_file_bits 
        << "\n print_bits            : " << m_print_bits
        << "\n";     
  }
}

//--------------------
// Print image shape parameters
void 
ImgPeakFinderAB::printShapeParameters()
{
  MsgLog(name(), debug, 
           "\n Rows    : " << m_rows   
        << "\n Columns : " << m_cols
        << "\n Size    : " << m_size
        << "\n"
	);
}
//--------------------
void 
ImgPeakFinderAB::setSelectionMode()
{
  m_sel_mode = SELECTION_OFF;
  if (m_sel_mode_str == "SELECTION_ON")  m_sel_mode = SELECTION_ON;
  if (m_sel_mode_str == "SELECTION_INV") m_sel_mode = SELECTION_INV;
}

//--------------------
/// Method which is called once at the beginning of the job
void 
ImgPeakFinderAB::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
ImgPeakFinderAB::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
ImgPeakFinderAB::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ImgPeakFinderAB::event(Event& evt, Env& env)
{
  m_count++;
  if(m_count == 1) init(evt, env);
  if( m_print_bits & 32 ) printEventRecord(evt);

  maskUpdateControl();
  resetForEventProcessing();
  if( m_print_bits & 512 ) printSelectionStatisticsByCounter();

  procData(evt);

  bool isSelected = eventSelector();

  if( m_print_bits &   4 ) printEventSelectionPars(evt, isSelected);
  if( m_print_bits & 128 ) printVectorOfPeaks();

  if (m_sel_mode == SELECTION_ON  && !isSelected) {skip(); return;}
  if (m_sel_mode == SELECTION_INV &&  isSelected) {skip(); return;}

  ++ m_count_selected;
  doOperationsForSelectedEvent(evt);
  if( m_print_bits & 1024 ) printEventRecord(evt,std::string(" selected"));

  if (m_sel_mode == SELECTION_OFF) return;
}

/// Method which is called at the end of the calibration cycle
void 
ImgPeakFinderAB::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ImgPeakFinderAB::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ImgPeakFinderAB::endJob(Event& evt, Env& env)
{
  if( m_out_file_bits & 1 ) saveImgArrayInFile<int16_t>( m_maskFile_out, m_mask );
  if( m_out_file_bits & 2 ) saveImgArrayInFile<float>  ( m_fracFile_out, m_frac_noisy_evts );
  //if( m_print_bits & 1024 ) 
  printJobSummary();  
}

//--------------------
void 
ImgPeakFinderAB::init(Event& evt, Env& env)
{
  defineImageShape(evt, m_str_src, m_key, m_shape);
    m_rows = m_shape[0];
    m_cols = m_shape[1];
    m_size = m_rows*m_cols;
    m_rows1= m_rows - 1;
    m_cols1= m_cols - 1;

    instArrays();

    resetStatArrays();
    getMaskFromFile();
    evaluateVectorOfIndexesForMedian();

    if( m_print_bits & 1  ) printInputParameters();
    if( m_print_bits & 1  ) printShapeParameters();
    if( m_print_bits & 2  ) printMaskStatistics();
    if( m_print_bits & 64 ) printVectorOfIndexesForMedian();
    if( m_print_bits & 64 ) printMatrixOfIndexesForMedian();

    m_time = new TimeInterval();
    m_time -> startTimeOnce();
}

//--------------------
void 
ImgPeakFinderAB::instArrays()
{
    m_stat            = new unsigned [m_size];
    m_mask            = new int16_t  [m_size];
    m_frac_noisy_evts = new float    [m_size];
    m_signal          = new double   [m_size];
    m_bkgd            = new double   [m_size];
    m_noise           = new double   [m_size];
    m_proc_status     = new uint16_t [m_size];
}

//--------------------
/// Reset arrays for statistics accumulation
void
ImgPeakFinderAB::resetStatArrays()
{
  std::fill_n(m_stat           , m_size, 0 );
  std::fill_n(m_frac_noisy_evts, m_size, 0.);
}

//--------------------
/// Reset signal arrays
void
ImgPeakFinderAB::resetSignalArrays()
{
  std::fill_n(m_signal     , m_size, double(0));
  std::fill_n(m_bkgd       , m_size, double(0));
  std::fill_n(m_noise      , m_size, double(0));
  std::fill_n(m_proc_status, m_size, 0);
}

//--------------------
/// Reset for event
void
ImgPeakFinderAB::resetForEventProcessing()
{
  v_peaks.clear(); // clear the vector of peaks
  v_peaks.reserve(1000);
  resetSignalArrays();
}

//--------------------
// This method decides when to re-evaluate the mask and call appropriate methods
void 
ImgPeakFinderAB::maskUpdateControl()
{
  if ( ++m_count_mask_update <  m_nevents_mask_update+1 ) return;
  if (   m_count_mask_update == m_nevents_mask_update+1 ) { 
     // Initialization for the mask re-evaluation

     if( m_print_bits & 16 ) cout << "Event " << m_count << " Start to collect data for mask re-evaluation cycle\n";
     resetStatArrays();
     m_count_mask_accum = 0;
  }

  if ( ++m_count_mask_accum < m_nevents_mask_accum ) return;
     // Re-evaluate the mask and reset counter

  if( m_print_bits & 16 ) cout << "Event " << m_count << " Stop to collect data for mask re-evaluation cycle, update the mask\n";
  procStatArrays();        // Process statistics, re-evaluate the mask
  m_count_mask_update = 0; // Reset counter
}

//--------------------
/// Process accumulated stat arrays and evaluate m_ave(rage) and m_rms arrays
void 
ImgPeakFinderAB::procStatArrays()
{
  unsigned long  npix_noisy = 0;
  unsigned long  npix_total = 0;
  
        for (unsigned r=0; r!=m_rows; ++r) {
          for (unsigned c=0; c!=m_cols; ++c) {
            unsigned i = r*m_cols + c;

            npix_total ++;
	    unsigned stat = m_stat[i];
	    
	    if(m_count_mask_accum > 0) { 
	      
	      float fraction_of_noisy_events = float(stat) / m_count_mask_accum; 

              m_frac_noisy_evts[i] = fraction_of_noisy_events;

	      if (fraction_of_noisy_events < m_frac_noisy_imgs) {

                m_mask[i] = 1; 
	      }
	      else
	      {
                m_mask[i] = 0; 
                npix_noisy ++;
	      }
            } 
          }
	}

    float fraction = (npix_total>0) ? float(npix_noisy)/npix_total : 0;
    if( m_print_bits & 2 ) MsgLog(name(), info, 
                                     "  Event:"                     << m_count
                                  << "  Collected for mask update:" << m_count_mask_accum 
                                  << "  Statistics: Nnoisy:"        << npix_noisy
				  << "  Ntotal:"                    << npix_total
                                  << "  Nnoisy/Ntotal pixels:"      << fraction );
}

//--------------------
// Process data in the event()
void 
ImgPeakFinderAB::procData(Event& evt)
{
  shared_ptr< ndarray<const double,2> > img = evt.get(m_str_src, m_key, &m_src);
  if (img.get()) {
    const double* m_data = img->data();

    collectStat(m_data);

    findPeaks();

    if(m_key_signal_out != "") {
        shared_ptr< ndarray<double,2> > img2d( new ndarray<double,2>(m_signal, m_shape) );
        evt.put(img2d, m_src, m_key_signal_out);
      }
  } 
  else
  {
    MsgLog(name(), info, "Image is not available in the event(...) for source:" << m_str_src << " key:" << m_key);
  }
}

//--------------------
/// Collect statistics in one section
/// Loop over one 2x1 section pixels, evaluate S/N and count statistics above threshold 
void 
ImgPeakFinderAB::collectStat(const double* data)
{
  for (unsigned r=0; r!=m_rows; ++r) {
    for (unsigned c=0; c!=m_cols; ++c) {
      unsigned i = r*m_cols + c;

      // 1) Apply the median algorithm to the pixel
      MedianResult median = evaluateSoNForPixel(r,c,data);

      // 2) Accumulate statistics of signal or noisy pixels
      if ( abs( median.SoN ) > m_SoNThr_noise ) m_stat[i] ++; 

      // 3) For masked array
      if (m_mask[i] != 0) {

        // 3a) produce signal/background/noise arrays
        m_signal[i] = median.sig; // for signal
        m_bkgd  [i] = median.avg; // for background
        m_noise [i] = median.rms; // for noise

        // 3b) Mark signal pixel for processing
        m_proc_status[i] = ( median.SoN > m_SoNThr_signal ) ? 255 : 0; 
      }
      else
      {
        m_signal     [i] = 0;
        m_bkgd       [i] = 0;
        m_noise      [i] = 0;
        m_proc_status[i] = 0;
      }
    }
  }
}

//--------------------
/// Find peaks in one section
/// Loop over one 2x1 section pixels and find the "connected" areas for peak region.
void 
ImgPeakFinderAB::findPeaks()
{
  for (unsigned r=0; r!=m_rows; ++r) {
    for (unsigned c=0; c!=m_cols; ++c) {
      unsigned i = r*m_cols + c;

      if( m_proc_status[i] & 1 ) {

	//cout << "Begin the new cluster, q,s,c,r=" << quad << " " << sect << " "  << ic << " " << ir << endl;

	// Initialization of the peak parameters
        PeakWork pw;
	pw.peak_npix       = 0;
	pw.peak_bkgd_tot   = 0;
	pw.peak_noise2_tot = 0;
	pw.peak_amp_tot    = 0;
	pw.peak_amp_max    = 0;
	pw.peak_amp_x_row1 = 0;
	pw.peak_amp_x_row2 = 0;
	pw.peak_amp_x_col1 = 0;
	pw.peak_amp_x_col2 = 0;

	// Begin to iterate over connected region
        // when it is done the connected region is formed
        iterateOverConnectedPixels(r,c,pw); 

	if( peakSelector(pw) ) savePeakInVector(pw);
      }
    }
  }
  if( m_print_bits & 2048) MsgLog(name(), info, "Number of selected peaks=" << v_peaks.size() << "\n");
}

//--------------------
// Flood-fill recursive iteration method in order to find the region of connected pixels
void 
ImgPeakFinderAB::iterateOverConnectedPixels(int ir, int ic, PeakWork& pw)
{
  unsigned i = ir*m_cols + ic;
  double  amp = m_signal[i];
  double noise= m_noise [i];

  pw.peak_npix       ++;
  pw.peak_bkgd_tot   +=  m_bkgd[i];
  pw.peak_noise2_tot +=  noise*noise; // sum the pixel noise quadratically, as randomly fluctuating.
  pw.peak_amp_tot    +=  amp;
  pw.peak_amp_x_row1 += (amp*ir);
  pw.peak_amp_x_row2 += (amp*ir*ir);
  pw.peak_amp_x_col1 += (amp*ic);
  pw.peak_amp_x_col2 += (amp*ic*ic);

  if (amp > pw.peak_amp_max) pw.peak_amp_max = amp;

  m_proc_status[i] ^= 1; // set the 1st bit to zero.

  if(ir+1 < m_rows1    && m_proc_status[i+m_cols] & 1 ) iterateOverConnectedPixels(ir+1, ic  , pw); 
  if(ic+1 < m_cols1    && m_proc_status[i+1]      & 1 ) iterateOverConnectedPixels(ir  , ic+1, pw); 
  if(ir-1 >=0          && m_proc_status[i-m_cols] & 1 ) iterateOverConnectedPixels(ir-1, ic  , pw); 
  if(ic-1 >=0          && m_proc_status[i-1]      & 1 ) iterateOverConnectedPixels(ir  , ic-1, pw); 
}


//--------------------
// Check the peak quality and return true for good peak
bool
ImgPeakFinderAB::peakSelector(PeakWork& pw) {

  if( m_print_bits & 2048 && pw.peak_npix > m_peak_npix_min-2) printPeakWork(pw);

  if(pw.peak_npix < m_peak_npix_min) return false;
  if(pw.peak_npix > m_peak_npix_max) return false;
  if(m_peak_amp_tot_thr > 1 && pw.peak_amp_tot < m_peak_amp_tot_thr) return false;   
  //if (pw.peak_amp_max < m_peak_amp_max_thr) return false;

  pw.peak_noise = std::sqrt( pw.peak_noise2_tot / pw.peak_npix );
  pw.peak_SoN = (pw.peak_noise > 0) ? pw.peak_amp_tot / pw.peak_noise : 0;
  if(pw.peak_SoN < m_peak_SoN_thr) return false;   

  return true;
}

//--------------------
// This is for test print only
void 
ImgPeakFinderAB::printPeakWork(PeakWork& pw) {

    double row    = pw.peak_amp_x_row1 / pw.peak_amp_tot; 
    double col    = pw.peak_amp_x_col1 / pw.peak_amp_tot;
    pw.peak_noise = std::sqrt( pw.peak_noise2_tot / pw.peak_npix );
    pw.peak_SoN   = (pw.peak_noise > 0) ? pw.peak_amp_tot / pw.peak_noise : 0;

    MsgLog(name(), info, 
             "Peak candidate: " 
          << "row, col="           << std::setprecision(1) << std::fixed << row
	  << ", "                 << col
          << "  npix/min:max="    << pw.peak_npix
          << "/"                  << m_peak_npix_min
          << ":"                  << m_peak_npix_max
	  << "  Btot="            << pw.peak_bkgd_tot
          << "  noise="           << pw.peak_noise
          << "  ampmax="          << pw.peak_amp_max
          << "  amptot/thr="      << pw.peak_amp_tot
          << "/"                  << m_peak_amp_tot_thr
          << "  SoN/thr="         << pw.peak_SoN
          << "/"                  << m_peak_SoN_thr << std::setprecision(6) 
	   );
}

//--------------------
// Creates, fills, and saves the object of structure Peak. 
void 
ImgPeakFinderAB::savePeakInVector(PeakWork& pw) {
  //MsgLog(name(), info, "Save peak info, npix =" << m_peak_npix << ", amp=" << m_peak_amp_tot;);
  Peak peak;
  peak.row       = pw.peak_amp_x_row1 / pw.peak_amp_tot; 
  peak.col       = pw.peak_amp_x_col1 / pw.peak_amp_tot;
  peak.sigma_row = std::sqrt( pw.peak_amp_x_row2/pw.peak_amp_tot - peak.row*peak.row );
  peak.sigma_col = std::sqrt( pw.peak_amp_x_col2/pw.peak_amp_tot - peak.col*peak.col );
  peak.ampmax    = pw.peak_amp_max;
  peak.amptot    = pw.peak_amp_tot;
  peak.bkgdtot   = pw.peak_bkgd_tot;
  peak.noise     = pw.peak_noise;
  peak.SoN       = pw.peak_SoN;
  peak.npix      = pw.peak_npix;

  v_peaks.push_back(peak);   
}

//--------------------
/// Print vector of peaks
void 
ImgPeakFinderAB::printVectorOfPeaks()
{
  MsgLog(name(), info, "Number of peaks in the event =" << v_peaks.size(););
  int i=0;
  for( vector<Peak>::const_iterator p  = v_peaks.begin();
                                    p != v_peaks.end(); p++ ) {
    MsgLog(name(), info, 
             "Peak:"       << ++i 
          << " col="       << p->col
          << " row="       << p->row
          << " npix="      << p->npix 
          << " SoN="       << p->SoN
          << " amptot="    << p->amptot
          << " noise="     << p->noise
          << " bkgdtot="   << p->bkgdtot
          << " ampmax="    << p->ampmax
          << " sigma_col=" << p->sigma_col
          << " sigma_row=" << p->sigma_row
	   );
  }
}

//--------------------
// Check the peak quality and return true for good peak
bool
ImgPeakFinderAB::eventSelector() {

  if (v_peaks.size() < m_event_npeak_min) return false;
  if (v_peaks.size() > m_event_npeak_max) return false;
  
  // Threshold on total ADC amplitude of all peaks is applied if m_event_amp_tot_thr is set > 1
 
  if ( m_event_amp_tot_thr > 1 ) {
    m_event_amp_tot = 0;
    for( vector<Peak>::const_iterator p  = v_peaks.begin();
                                      p != v_peaks.end(); p++ ) {
      m_event_amp_tot += p->amptot;
    }
    if (m_event_amp_tot < m_event_amp_tot_thr) return false;
  }

  return true;
}

//--------------------
/// Print parameters used in the event selection
void 
ImgPeakFinderAB::printEventSelectionPars(Event& evt, bool isSelected)
{
    MsgLog(name(), info, 
	   //"  Run="         << stringRunNumber(evt) 
              "Event="        << stringFromUint(m_count)  
           << " "             << stringTimeStamp(evt)
           << " mode="        << m_sel_mode_str  
           << " N peaks/min=" << v_peaks.size()
           << " / "           << m_event_npeak_min
           << " A tot/thr="   << m_event_amp_tot
           << " / "           << m_event_amp_tot_thr
           << " isSelected="  << isSelected  
	   );
}

//--------------------
/// Evaluate vector of indexes for mediane algorithm
/// The area of pixels for the mediane algorithm is defined as a ring from m_rmin to m_rmin + m_dr
void 
ImgPeakFinderAB::evaluateVectorOfIndexesForMedian()
{
  v_indForMediane.clear();

  TwoIndexes inds;
  int indmax = int(m_rmin + m_dr);
  int indmin = -indmax;

  for (int i = indmin; i <= indmax; ++ i) {
    for (int j = indmin; j <= indmax; ++ j) {

      float r = std::sqrt( float(i*i + j*j) );
      if ( r < m_rmin || r > m_rmin + m_dr ) continue;
      inds.i = i;
      inds.j = j;
      v_indForMediane.push_back(inds);
    }
  }
}

//--------------------

void 
ImgPeakFinderAB::printMatrixOfIndexesForMedian()
{
  int indmax = int(m_rmin + m_dr);
  int indmin = -indmax;

  cout << "ImgPeakFinderAB::printMatrixOfIndexesForMedian():" << endl;
  for (int i = indmin; i <= indmax; ++ i) {
    for (int j = indmin; j <= indmax; ++ j) {

      float r = std::sqrt( float(i*i + j*j) );
      int status = ( r < m_rmin || r > m_rmin + m_dr ) ? 0 : 1;
      if (i==0 && j==0) cout << " +";
      else              cout << " " << status;
    }
    cout << endl;
  }
}

//--------------------
/// Print vector of indexes for mediane algorithm
void 
ImgPeakFinderAB::printVectorOfIndexesForMedian()
{
  std::cout << "ImgPeakFinderAB::printVectorOfIndexesForMedian():" << std::endl;
  int n_pairs_in_line=0;
  for( vector<TwoIndexes>::const_iterator ij  = v_indForMediane.begin();
                                          ij != v_indForMediane.end(); ij++ ) {

    cout << " (" << ij->i << "," << ij->j << ")";
    if ( ++n_pairs_in_line > 9 ) {cout << "\n"; n_pairs_in_line=0;}
  }   
  cout << "\nVector size: " << v_indForMediane.size() << endl;
}

//--------------------
/// Apply median algorithm for one pixel
MedianResult
ImgPeakFinderAB::evaluateSoNForPixel(unsigned row, unsigned col, const double* data)
{

  unsigned sum0 = 0;
  double   sum1 = 0;
  double   sum2 = 0;

  for( vector<TwoIndexes>::const_iterator ij  = v_indForMediane.begin();
                                          ij != v_indForMediane.end(); ij++ ) {
    int ic = col + (ij->i);
    int ir = row + (ij->j);

    if(ic < 0)       continue;
    if(ic > m_cols1) continue;
    if(ir < 0)       continue;
    if(ir > m_rows1) continue;

    double  amp = data[ir*m_cols + ic];
    sum0 ++;
    sum1 += amp;
    sum2 += amp*amp;
  }

  MedianResult res = {0,0,0,0};

  if ( sum0 > 0 ) {
    res.avg = sum1/sum0;                                // Averaged background level
    res.rms = std::sqrt( sum2/sum0 - res.avg*res.avg ); // RMS os the background around peak
    double dat = data[row*m_cols + col];
    double dif = dat - res.avg;
    res.sig = (dat>dif) ? dif : dat;                    // Signal above the background
    if (res.rms>0) res.SoN = res.sig/res.rms;           // S/N ratio
  }

  return res;
}

//--------------------

void 
ImgPeakFinderAB::printEventRecord(Event& evt, std::string comment)
{
  MsgLog( name(), info,  "Run="    << stringRunNumber(evt) 
                     << " Evt="    << stringFromUint(m_count) 
                     << " Sel="    << stringFromUint(m_count_selected) 
                     << " Time="   << stringTimeStamp(evt) 
	             << comment.c_str()
  );
}

//--------------------

void
ImgPeakFinderAB::getMaskFromFile()
{
  if (m_maskFile_inp != std::string("")) {
     MsgLog( name(), info, " Use initial hot pixel mask from file:" << m_maskFile_inp.c_str() ); 
     m_mask_initial = new ImgAlgos::ImgParametersV1(m_maskFile_inp);
     ImgAlgos::ImgParametersV1::pars_t* mask_data = m_mask_initial->data();
     for (unsigned i=0; i<m_size; i++) m_mask[i] = (int16_t)mask_data[i]; // do not use memcpy because of type conversion
  }
  else
  {
     MsgLog( name(), info, " Use default initial hot pixel mask made of units." ); 
     m_mask_initial = new ImgAlgos::ImgParametersV1(m_shape,1); 
     std::fill_n(m_mask, (int)m_size, 1);
  }
  if(m_print_bits & 2) m_mask_initial -> print("Hot pixel initial mask");
}

//--------------------

void 
ImgPeakFinderAB::printMaskStatistics()
{
  int Nof0=0;
  int Nof1=0;

  for (unsigned i=0; i<m_size; i++) {
      if (m_mask[i] == 0) Nof0++;
      if (m_mask[i] == 1) Nof1++;
  }

  MsgLog(name(), info, "Mask statistics: Nof0: " << Nof0 
                                    << " Nof1: " << Nof1
                                    << " Ntot: " << m_size
                                    << " Nof0 / Ntot = " << float(Nof0)/m_size );
}

//--------------------

void 
ImgPeakFinderAB::doOperationsForSelectedEvent(Event& evt)
{
  // Define the file name
  std::string fname = m_evtFile_out 
                    + "r"  + stringRunNumber(evt) 
                    + "-e" + stringFromUint(m_count) 
                    + "-"  + stringTimeStamp(evt);
  std::string fname_arr   =  fname + ".txt";
  std::string fname_peaks =  fname + "-peaks.txt";

  if( m_out_file_bits & 4 ) saveImgArrayInFile<double> (fname_arr, m_signal);
  if( m_out_file_bits & 8 ) savePeaksInFile (fname_peaks, v_peaks);

  savePeaksInEvent(evt);
}

//--------------------

/// Save 4-d array of CSPad structure in file
template <typename T>
void 
ImgPeakFinderAB::saveImgArrayInFile(const std::string& fname, const T* arr)
{  
  if (not fname.empty()) {
    if( m_print_bits & 8 ) MsgLog(name(), info, "Save 2-d image array in file " << fname.c_str());
    std::ofstream out(fname.c_str());
        for (unsigned r=0; r!=m_rows; ++r) {
          for (unsigned c=0; c!=m_cols; ++c) {

            out << arr[r*m_cols + c] << ' ';
          }
          out << '\n';
        }
    out.close();
  }
}


//--------------------
// Save vector of peaks in the event
void 
ImgPeakFinderAB::savePeaksInFile (std::string& fname, std::vector<Peak> peaks)
{
  if( m_print_bits & 256 ) printVectorOfPeaks();

  ofstream file; 
  file.open(fname.c_str(),ios_base::out);

  for( vector<Peak>::const_iterator itv  = peaks.begin();
                                    itv != peaks.end(); itv++ ) {
    file << itv->col       << "  "
         << itv->row       << "  "
         << itv->npix      << "  "      
         << itv->sigma_col << "  "
         << itv->sigma_row << "  "
         << itv->ampmax    << "  "
         << itv->amptot    << endl; 
  }

  file.close();
}

//--------------------
// Save vector of peaks in the event
void 
ImgPeakFinderAB::savePeaksInEvent(Event& evt)
{
  shared_ptr< std::vector<Peak> >  sppeaks( new std::vector<Peak>(v_peaks) );
  if( v_peaks.size() > 0 ) evt.put(sppeaks, m_src, m_key_peaks_out);
}

//--------------------
// Print current selection statistics
void 
ImgPeakFinderAB::printSelectionStatisticsByCounter() // Event& evt)
{
    if (  m_count < 5 
      || (m_count < 50   && m_count % 10   == 0)
      || (m_count < 500  && m_count % 100  == 0)
      || (m_count % 1000 == 0)
	) printSelectionStatistics();
}

//--------------------
// Print current selection statistics
void 
ImgPeakFinderAB::printSelectionStatistics() // Event& evt)
{
  float fraction = (m_count > 0) ? 100.*float(m_count_selected) / m_count : 0;
  double dt = m_time -> getCurrentTimeInterval();
  float rate = (dt > 0) ? float(m_count) / dt : 0;

  std::stringstream sc; sc.setf(ios_base::fixed); sc.width(7);  sc.fill(' '); sc << m_count;
  std::stringstream sh; sh.setf(ios_base::fixed); sh.width(7);  sh.fill(' '); sh << m_count_selected;
  std::stringstream sf; sf.setf(ios_base::fixed); sf.width(6);  sf.fill(' '); sf.precision(2); sf << fraction;
  std::stringstream st; st.setf(ios_base::fixed); st.width(11); st.fill(' '); st.precision(3); st << dt;
  std::stringstream sr; sr.setf(ios_base::fixed); sr.width(6);  sr.fill(' '); sr.precision(3); sr << rate;
  std::string s = "NFrames:" + sc.str()
	        +  " NHits:" + sh.str() + " ("  + sf.str() + "%)" 
	        +   " Time:" + st.str() + "sec (" + sr.str() + "fps)";

  MsgLog( name(), info, s );
}

//--------------------
// Print current selection statistics
void 
ImgPeakFinderAB::printJobSummary()
{
  MsgLog( name(), info, "===== JOB SUMMARY =====" );  
  printSelectionStatistics();
  m_time -> stopTime(m_count);
}

//--------------------

} // namespace ImgAlgos

//--------------------
