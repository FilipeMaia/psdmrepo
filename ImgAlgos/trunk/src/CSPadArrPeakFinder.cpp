//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadArrPeakFinder...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/CSPadArrPeakFinder.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <fstream>
#include <cmath>
#include <iomanip> // for setw, setfill
#include <sstream> // for streamstring

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
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(CSPadArrPeakFinder)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------

CSPadArrPeakFinder::CSPadArrPeakFinder (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key()
  , m_key_peaks_out()  
  , m_maskFile_inp()
  , m_maskFile_out()
  , m_fracFile_out()
  , m_evtFile_out()
  , m_rmin()
  , m_dr()
  , m_SoNThr()
  , m_frac_noisy_imgs()
  , m_peak_npix_min()
  , m_peak_npix_max()
  , m_peak_amp_tot_thr()
  , m_event_npeak_min()
  , m_event_amp_tot_thr()
  , m_nevents_mask_update()
  , m_nevents_mask_accum()
  , m_sel_mode_str()
  , m_out_file_bits()
  , m_print_bits()
  , m_count(0)
  , m_count_mask_update(0)
  , m_count_mask_accum(0)
{
  // get the values from configuration or use defaults
  m_str_src           = configStr("source",     "DetInfo(:Cspad)");
  m_key               = configStr("key",        "");                 //"calibrated"
  m_key_peaks_out     = configStr("key_peaks_out", "peaks");
  m_maskFile_inp      = configStr("hot_pix_mask_inp_file", "cspad-pix-mask-in.dat");
  m_maskFile_out      = configStr("hot_pix_mask_out_file", "cspad-pix-mask-out.dat");
  m_fracFile_out      = configStr("frac_noisy_evts_file",  "cspad-pix-frac-out.dat");
  m_evtFile_out       = configStr("evt_file_out",          "./cspad-ev-");
  m_rmin              = config   ("rmin",              3);
  m_dr                = config   ("dr",                1);
  m_SoNThr            = config   ("SoNThr",            3);
  m_frac_noisy_imgs   = config   ("frac_noisy_imgs", 0.1); 

  m_peak_npix_min     = config   ("peak_npix_min",         4);
  m_peak_npix_max     = config   ("peak_npix_max",        25);
  m_peak_amp_tot_thr  = config   ("peak_amp_tot_thr",   100.);

  m_event_npeak_min   = config   ("event_npeak_min",      10);
  m_event_amp_tot_thr = config   ("event_amp_tot_thr", 1000.);

  m_nevents_mask_update= config   ("nevents_mask_update",   100);
  m_nevents_mask_accum = config   ("nevents_mask_accum",     50);

  m_sel_mode_str      = configStr("selection_mode", "SELECTION_ON");
  m_out_file_bits     = config   ("out_file_bits",         0); 
  m_print_bits        = config   ("print_bits",            0);

  // initialize arrays
  std::fill_n(&m_segMask[0], int(MaxQuads), 0U);

  setSelectionMode(); // m_sel_mode_str -> enum m_sel_mode 
  resetStatArrays();
  resetSignalArrays();
  getMaskFromFile(); // load the initial hot-pixel mask from file or default 
  if( m_print_bits & 1 ) printMaskStatistics();
}

//--------------
// Destructor --
//--------------
CSPadArrPeakFinder::~CSPadArrPeakFinder ()
{
}
//--------------------

// Print input parameters
void 
CSPadArrPeakFinder::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source                : " << m_str_src
        << "\n key                   : " << m_key      
        << "\n m_key_peaks_out       : " << m_key_peaks_out 
        << "\n m_maskFile_inp        : " << m_maskFile_inp    
        << "\n m_maskFile_out        : " << m_maskFile_out    
        << "\n m_fracFile_out        : " << m_fracFile_out    
        << "\n m_evtFile_out         : " << m_evtFile_out  
        << "\n m_rmin                : " << m_rmin    
        << "\n m_dr                  : " << m_dr     
        << "\n m_SoNThr              : " << m_SoNThr     
        << "\n m_frac_noisy_imgs     : " << m_frac_noisy_imgs    
        << "\n m_peak_npix_min       : " << m_peak_npix_min     
        << "\n m_peak_npix_max       : " << m_peak_npix_max     
        << "\n m_peak_amp_tot_thr    : " << m_peak_amp_tot_thr  
        << "\n m_event_npeak_min     : " << m_event_npeak_min   
        << "\n m_event_amp_tot_thr   : " << m_event_amp_tot_thr 
        << "\n m_nevents_mask_update : " << m_nevents_mask_update
        << "\n m_nevents_mask_accum  : " << m_nevents_mask_accum 
        << "\n m_sel_mode_str        : " << m_sel_mode_str 
        << "\n m_sel_mode            : " << m_sel_mode 
        << "\n m_out_file_bits       : " << m_out_file_bits 
        << "\n print_bits            : " << m_print_bits
        << "\n";     

    log << "\n MaxQuads   : " << MaxQuads    
        << "\n MaxSectors : " << MaxSectors  
        << "\n NumColumns : " << NumColumns  
        << "\n NumRows    : " << NumRows     
        << "\n SectorSize : " << SectorSize  
        << "\n";
  }
}

/// Method which is called once at the beginning of the job
void 
CSPadArrPeakFinder::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();

  evaluateVectorOfIndexesForMedian();
  if( m_print_bits & 64 ) printVectorOfIndexesForMedian();
  if( m_print_bits & 64 ) printMatrixOfIndexesForMedian();

}

/// Method which is called at the beginning of the run
void 
CSPadArrPeakFinder::beginRun(Event& evt, Env& env)
{
  // Find all configuration objects matching the source address
  // provided in configuration. If there is more than one configuration 
  // object is found then complain and stop.
  
  //std::string src = configStr("source", "DetInfo(:Cspad)");
  int count = 0;
  
  // need to know segment mask which is availabale in configuration only
  shared_ptr<Psana::CsPad::ConfigV1> config1 = env.configStore().get(m_str_src, &m_src);
  if (config1.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config1->asicMask()==1 ? 0x3 : 0xff; }
    ++ count;
  }
  
  shared_ptr<Psana::CsPad::ConfigV2> config2 = env.configStore().get(m_str_src, &m_src);
  if (config2.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config2->roiMask(i); }
    ++ count;
  }

  shared_ptr<Psana::CsPad::ConfigV3> config3 = env.configStore().get(m_str_src, &m_src);
  if (config3.get()) {
    for (int i = 0; i < MaxQuads; ++i) { m_segMask[i] = config3->roiMask(i); }
    ++ count;
  }

  if (not count) {
    MsgLog(name(), error, "No CSPad configuration objects found. Terminating.");
    terminate();
    return;
  }
  
  if (count > 1) {
    MsgLog(name(), error, "Multiple CSPad configuration objects found, use more specific source address. Terminating.");
    terminate();
    return;
  }

  MsgLog(name(), info, "Found CSPad object with address " << m_src);
  if (m_src.level() != Pds::Level::Source) {
    MsgLog(name(), error, "Found CSPad configuration object with address not at Source level. Terminating.");
    terminate();
    return;
  }

  const Pds::DetInfo& dinfo = static_cast<const Pds::DetInfo&>(m_src);
  // validate that this is indeed CSPad, should always be true, but
  // additional protection here should not hurt
  if (dinfo.device() != Pds::DetInfo::Cspad) {
    MsgLog(name(), error, "Found CSPad configuration object with invalid address. Terminating.");
    terminate();
    return;
  }
}



/// Method which is called at the beginning of the calibration cycle
void 
CSPadArrPeakFinder::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadArrPeakFinder::event(Event& evt, Env& env)
{
  maskUpdateControl();

  resetForEventProcessing();

  shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(m_str_src, m_key, &m_src);
  if (data1.get()) {

    ++ m_count;
    //setCollectionMode();
    
    int nQuads = data1->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {

      const CsPad::ElementV1& quad = data1->quads(iq);
      const ndarray<int16_t, 3>& data = quad.data();
      collectStatInQuad(quad.quad(), data.data());
    }    
  }
  
  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_str_src, m_key, &m_src);
  if (data2.get()) {

    ++ m_count;
    //setCollectionMode();
    
    int nQuads = data2->quads_shape()[0];
    for (int iq = 0; iq != nQuads; ++ iq) {
      
      const CsPad::ElementV2& quad = data2->quads(iq);
      const ndarray<int16_t, 3>& data = quad.data();
      collectStatInQuad(quad.quad(), data.data());
    } 
  }

  if( m_print_bits &  32 ) printTimeStamp(evt);
  if( m_print_bits & 128 ) printVectorOfPeaks();

  if (m_sel_mode == SELECTION_ON  && !eventSelector()) {skip(); return;}
  if (m_sel_mode == SELECTION_INV &&  eventSelector()) {skip(); return;}

  doOperationsForSelectedEvents(evt);

  if (m_sel_mode == SELECTION_OFF) return;
}


/// Method which is called at the end of the calibration cycle
void 
CSPadArrPeakFinder::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
CSPadArrPeakFinder::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
CSPadArrPeakFinder::endJob(Event& evt, Env& env)
{
  if( m_out_file_bits & 1 ) saveCSPadArrayInFile<int16_t>( m_maskFile_out, m_mask );
  if( m_out_file_bits & 2 ) saveCSPadArrayInFile<float>  ( m_fracFile_out, m_frac_noisy_evts );
}

//--------------------

void 
CSPadArrPeakFinder::setSelectionMode()
{
  m_sel_mode = SELECTION_OFF;
  if (m_sel_mode_str == "SELECTION_ON")  m_sel_mode = SELECTION_ON;
  if (m_sel_mode_str == "SELECTION_INV") m_sel_mode = SELECTION_INV;
}

//--------------------
// This method decides when to re-evaluate the mask and call appropriate methods
void 
CSPadArrPeakFinder::maskUpdateControl()
{
  if ( ++m_count_mask_update <  m_nevents_mask_update ) return;
  if (   m_count_mask_update == m_nevents_mask_update ) { 
     // Initialization for the mask re-evaluation

     if( m_print_bits & 16 ) cout << "Event " << m_count << " Start to collect data for mask re-evaluation cycle\n";
     resetStatArrays();
     m_count_mask_accum = 0;
  }

  //cout << "Event " << m_count << " is taken for mask re-evaluation\n";

  if ( ++m_count_mask_accum < m_nevents_mask_accum ) return;
     // Re-evaluate the mask and reset counter

  if( m_print_bits & 16 ) cout << "Event " << m_count << " Stop to collect data for mask re-evaluation cycle, update the mask\n";
  procStatArrays();        // Process statistics, re-evaluate the mask
  m_count_mask_update = 0; // Reset counter
}

//--------------------

/// Process accumulated stat arrays and evaluate m_ave(rage) and m_rms arrays
void 
CSPadArrPeakFinder::procStatArrays()
{
  if( m_print_bits & 4 ) MsgLog(name(), info, "Process statistics for collected total " << m_count_mask_accum << " events");

  unsigned long  npix_noisy = 0;
  unsigned long  npix_total = 0;
  
    for (int iq = 0; iq != MaxQuads; ++ iq) {
      for (int is = 0; is != MaxSectors; ++ is) {
        for (int ic = 0; ic != NumColumns; ++ ic) {
          for (int ir = 0; ir != NumRows; ++ ir) {

            npix_total ++;
	    unsigned stat = m_stat[iq][is][ic][ir];
	    
	    if(m_count_mask_accum > 0) { 
	      
	      float fraction_of_noisy_events = float(stat) / m_count_mask_accum; 

              m_frac_noisy_evts[iq][is][ic][ir] = fraction_of_noisy_events;

	      if (fraction_of_noisy_events < m_frac_noisy_imgs) {

                m_mask[iq][is][ic][ir] = 1; 
	      }
	      else
	      {
                npix_noisy ++;
	      }
	      
              //if (stat > 0) cout << "q,s,c,r=" << iq << " " << is << " " << ic << " " << ir
	      //                 << " stat, total=" << stat << " " << m_count << endl;

            } 
          }
        }
      }
    }
    cout << "Nnoisy, Ntotal, Nnoisy/Ntotal pixels =" << npix_noisy << " " << npix_total  << " " << double(npix_noisy)/npix_total << endl;
}

//--------------------
/// Reset for event
void
CSPadArrPeakFinder::resetForEventProcessing()
{
  v_peaks.clear(); // clear the vector of peaks
  resetSignalArrays();
}

//--------------------
/// Reset arrays for statistics accumulation
void
CSPadArrPeakFinder::resetStatArrays()
{
  std::fill_n(&m_stat           [0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0 );
  std::fill_n(&m_frac_noisy_evts[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0.);
}

//--------------------
/// Reset the dynamic mask of noisy pixels 
void
CSPadArrPeakFinder::resetMaskOfNoisyPix()
{
  std::fill_n(&m_mask[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 1);
}

//--------------------
/// Reset signal arrays
void
CSPadArrPeakFinder::resetSignalArrays()
{
  std::fill_n(&m_signal     [0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0);
  std::fill_n(&m_proc_status[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0);
}

//--------------------
/// Collect statistics
/// Loop over all 2x1 sections available in the event 
void 
CSPadArrPeakFinder::collectStatInQuad(unsigned quad, const int16_t* data)
{
  //cout << "collectStat for quad =" << quad << endl;

  int ind_in_arr = 0;
  for (unsigned sect = 0; sect < MaxSectors; ++ sect) {
    if (m_segMask[quad] & (1 << sect)) {
     
      const int16_t* sectData = data + ind_in_arr*SectorSize;

      collectStatInSect(quad, sect, sectData);

      findPeaksInSect  (quad, sect);

      ++ind_in_arr;
    }
  }
}

//--------------------
/// Collect statistics in one section
/// Loop over one 2x1 section pixels, evaluate S/N and count statistics above threshold 
void 
CSPadArrPeakFinder::collectStatInSect(unsigned quad, unsigned sect, const int16_t* sectData)
{
  for (int ic = 0; ic != NumColumns; ++ ic) {
    for (int ir = 0; ir != NumRows; ++ ir) {

      // 1) Apply the median algorithm to the pixel
      MedianResult median = evaluateSoNForPixel(ic,ir,sectData);

      // 2) Accumulate statistics of signal or noisy pixels
      if ( abs( median.SoN ) > m_SoNThr ) m_stat[quad][sect][ic][ir] ++; 

      // 3) For masked array
      if (m_mask[quad][sect][ic][ir] != 0) {

        // 3a) produce signal array
        m_signal[quad][sect][ic][ir] = int16_t (median.sig); // for signal
        //m_signal[quad][sect][ic][ir] = int16_t (median.avg); // for background - test only

        // 3b) Mark signal pixel for processing
        m_proc_status[quad][sect][ic][ir] = ( median.SoN > m_SoNThr ) ? 255 : 0; 
      }
      else
      {
        m_signal     [quad][sect][ic][ir] = 0;
        m_proc_status[quad][sect][ic][ir] = 0;
      }
    }
  }
}

//--------------------
/// Find peaks in one section
/// Loop over one 2x1 section pixels and find the "connected" areas for peak region.
void 
CSPadArrPeakFinder::findPeaksInSect(unsigned quad, unsigned sect)
{
  m_quad = quad;
  m_sect = sect;
  for (int ic = 0; ic != NumColumns; ++ ic) {
    for (int ir = 0; ir != NumRows; ++ ir) {

      if( m_proc_status[quad][sect][ic][ir] & 1 ) {

	//cout << "Begin the new cluster, q,s,c,r=" << m_quad << " " << m_sect << " "  << ic << " " << ir << endl;

	// Initialization of the peak parameters
	m_peak_npix       = 0;
	m_peak_amp_max    = 0;
	m_peak_amp_tot    = 0;
	m_peak_amp_x_col1 = 0;
	m_peak_amp_x_col2 = 0;
	m_peak_amp_x_row1 = 0;
	m_peak_amp_x_row2 = 0;

	// Begin to iterate over connected region
        // when it is done the connected region is formed
        iterateOverConnectedPixels(ic,ir); 

	if( peakSelector() ) savePeakInfo();
      }
    }
  }
}

//--------------------
// Flood-fill recursive iteration method in order to find the region of connected pixels
void 
CSPadArrPeakFinder::iterateOverConnectedPixels(int ic, int ir)
{
  int16_t amp = m_signal[m_quad][m_sect][ic][ir];

  m_peak_npix       ++;
  m_peak_amp_tot    +=  amp;
  m_peak_amp_x_col1 += (amp*ic);
  m_peak_amp_x_col2 += (amp*ic*ic);
  m_peak_amp_x_row1 += (amp*ir);
  m_peak_amp_x_row2 += (amp*ir*ir);
  if (amp > m_peak_amp_max) m_peak_amp_max = amp;

  m_proc_status[m_quad][m_sect][ic][ir] ^= 1; // set the 1st bit to zero.

  if(ir+1 < NumRows    && m_proc_status[m_quad][m_sect][ic][ir+1] & 1 ) iterateOverConnectedPixels(ic,   ir+1); 
  if(ic+1 < NumColumns && m_proc_status[m_quad][m_sect][ic+1][ir] & 1 ) iterateOverConnectedPixels(ic+1, ir  ); 
  if(ir-1 >=0          && m_proc_status[m_quad][m_sect][ic][ir-1] & 1 ) iterateOverConnectedPixels(ic,   ir-1); 
  if(ic-1 >=0          && m_proc_status[m_quad][m_sect][ic-1][ir] & 1 ) iterateOverConnectedPixels(ic-1, ir  ); 
}


//--------------------
// Check the peak quality and return true for good peak
bool
CSPadArrPeakFinder::peakSelector() {

  if (m_peak_npix < m_peak_npix_min)       return false;
  if (m_peak_npix > m_peak_npix_max)       return false;
  if (m_peak_amp_tot < m_peak_amp_tot_thr) return false;  
  //if (m_peak_amp_max < m_peak_amp_max_thr) return false;
  return true;
}

//--------------------
// Creates, fills, and saves the object of structure Peak. 
void 
CSPadArrPeakFinder::savePeakInfo() {
  //MsgLog(name(), info, "Save peak info, npix =" << m_peak_npix << ", amp=" << m_peak_amp_tot;);
  Peak peak;
  peak.quad      = m_quad;
  peak.sect      = m_sect;
  peak.col       = m_peak_amp_x_col1 / m_peak_amp_tot;
  peak.row       = m_peak_amp_x_row1 / m_peak_amp_tot; 
  peak.sigma_col = std::sqrt( m_peak_amp_x_col2/m_peak_amp_tot - peak.col*peak.col );
  peak.sigma_row = std::sqrt( m_peak_amp_x_row2/m_peak_amp_tot - peak.row*peak.row );
  peak.ampmax    = m_peak_amp_max;
  peak.amptot    = m_peak_amp_tot;
  peak.npix      = m_peak_npix;

  v_peaks.push_back(peak);   
}

//--------------------
/// Print vector of peaks
void 
CSPadArrPeakFinder::printVectorOfPeaks()
{
  MsgLog(name(), info, "Number of peaks in the event =" << v_peaks.size(););
  int i=0;
  for( vector<Peak>::const_iterator p  = v_peaks.begin();
                                    p != v_peaks.end(); p++ ) {

    cout  << "  peak:"      << ++i 
          << "  quad="      << p->quad
          << "  sect="      << p->sect
          << "  col="       << p->col
          << "  row="       << p->row
          << "  npix="      << p->npix 
          << "  amptot="    << p->amptot
          << "  ampmax="    << p->ampmax
          << "  sigma_col=" << p->sigma_col
          << "  sigma_row=" << p->sigma_row
          << endl;
  }
}

//--------------------
// Check the peak quality and return true for good peak
bool
CSPadArrPeakFinder::eventSelector() {

  if (v_peaks.size() < m_event_npeak_min) return false;

  m_event_amp_tot = 0;
  for( vector<Peak>::const_iterator p  = v_peaks.begin();
                                    p != v_peaks.end(); p++ ) {
    m_event_amp_tot += p->amptot;
  }

  if (m_event_amp_tot < m_event_amp_tot_thr) return false;

  return true;
}


//--------------------
/// Evaluate vector of indexes for mediane algorithm
/// The area of pixels for the mediane algorithm is defined as a ring from m_rmin to m_rmin + m_dr
void 
CSPadArrPeakFinder::evaluateVectorOfIndexesForMedian()
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
CSPadArrPeakFinder::printMatrixOfIndexesForMedian()
{
  int indmax = int(m_rmin + m_dr);
  int indmin = -indmax;

  cout << "CSPadArrPeakFinder::printMatrixOfIndexesForMedian():" << endl;
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
CSPadArrPeakFinder::printVectorOfIndexesForMedian()
{
  std::cout << "CSPadArrPeakFinder::printVectorOfIndexesForMedian():" << std::endl;
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
CSPadArrPeakFinder::evaluateSoNForPixel(unsigned col, unsigned row, const int16_t* sectData)
{

  unsigned sum0 = 0;
  double   sum1 = 0;
  double   sum2 = 0;

  for( vector<TwoIndexes>::const_iterator ij  = v_indForMediane.begin();
                                          ij != v_indForMediane.end(); ij++ ) {
    int ic = col + (ij->i);
    int ir = row + (ij->j);

    if(ic < 0)           continue;
    if(ic > NumColumns1) continue;
    if(ir < 0)           continue;
    if(ir > NumRows1)    continue;

    double  amp = sectData[ir + ic*NumRows];
    sum0 ++;
    sum1 += amp;
    sum2 += amp*amp;
  }

  MedianResult res = {0,0,0,0};

  if ( sum0 > 0 ) {
    res.avg = sum1/sum0;                                // Averaged background level
    res.rms = std::sqrt( sum2/sum0 - res.avg*res.avg ); // RMS os the background around peak
    res.sig = sectData[row + col*NumRows] - res.avg;    // Signal above the background
    if (res.rms>0) res.SoN = res.sig/res.rms;        // S/N ratio
  }

  return res;
}

//--------------------

void 
CSPadArrPeakFinder::printEventId(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    MsgLog( name(), info, "Event="  << m_count << " ID: " << *eventId);
  }
}

//--------------------

void 
CSPadArrPeakFinder::printTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {

    MsgLog( name(), info, " Run="   <<  eventId->run()
                       << " Event=" <<  m_count 
                       << " Time="  <<  eventId->time() );
  }
}

//--------------------

void
CSPadArrPeakFinder::getMaskFromFile()
{
  if (m_maskFile_inp.c_str() != "") {
     m_mask_initial = new ImgAlgos::CSPadMaskV1(m_maskFile_inp);

     CSPadMaskV1::mask_t* mask_initial = m_mask_initial->getMask();
     int16_t* mask = &m_mask[0][0][0][0];
     for (int i=0; i<SIZE_OF_ARRAY; i++) mask[i] = mask_initial[i];
  }
  else
  {
     m_mask_initial = new ImgAlgos::CSPadMaskV1(1);
     resetMaskOfNoisyPix();
  }
}

//--------------------

void 
CSPadArrPeakFinder::printMaskStatistics()
{
  m_mask_initial -> printMaskStatistics(); 
}

//--------------------

std::string
CSPadArrPeakFinder::strEventCounter()
{
  std::stringstream ssEvNum; ssEvNum << std::setw(6) << std::setfill('0') << m_count;
  return ssEvNum.str();
}

//--------------------

std::string  
CSPadArrPeakFinder::strTimeStamp(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {

    //m_time = eventId->time();
    //std::stringstream ss;
    //ss << hex << t_msec;
    //string hex_msec = ss.str();

    return (eventId->time()).asStringFormat( "%Y-%m-%d-%H%M%S%f"); // "%Y-%m-%d %H:%M:%S%f%z"
  }
  else
    return std::string("time-stamp-is-not-defined");
}

//--------------------

std::string  
CSPadArrPeakFinder::strRunNumber(Event& evt)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    std::stringstream ssRunNum; ssRunNum << "r" << std::setw(4) << std::setfill('0') << eventId->run();
    return ssRunNum.str();
  }
  else
    return std::string("run-is-not-defined");
}

//--------------------

void 
CSPadArrPeakFinder::doOperationsForSelectedEvents(Event& evt)
{

  // Define the file name
  std::string fname = m_evtFile_out 
                    + strEventCounter() 
                    + "-" + strRunNumber(evt) 
                    + "-" + strTimeStamp(evt) + ".txt";

  if( m_out_file_bits & 4 ) saveCSPadArrayInFile<int16_t>  (fname, m_signal);

  savePeaksInEvent(evt);
  //saveSignalArrInEvent(evt); // not implemented yet
}

//--------------------

/// Save 4-d array of CSPad structure in file
template <typename T>
void 
CSPadArrPeakFinder::saveCSPadArrayInFile(std::string& fname, T arr[MaxQuads][MaxSectors][NumColumns][NumRows])
{  
  if (not fname.empty()) {
    if( m_print_bits & 8 ) MsgLog(name(), info, "Save CSPad-shaped array in file " << fname.c_str());
    std::ofstream out(fname.c_str());
    for (int iq = 0; iq != MaxQuads; ++ iq) {
      for (int is = 0; is != MaxSectors; ++ is) {
        for (int ic = 0; ic != NumColumns; ++ ic) {
          for (int ir = 0; ir != NumRows; ++ ir) {

            out << arr[iq][is][ic][ir] << ' ';
          }
          out << '\n';
        }
      }
    }
    out.close();
  }
}


//--------------------
// Save vector of peaks in the event
void 
CSPadArrPeakFinder::savePeaksInEvent(Event& evt)
{
  shared_ptr< std::vector<Peak> >  sppeaks( new std::vector<Peak>(v_peaks) );
  if( v_peaks.size() > 0 ) evt.put(sppeaks, m_src, m_key_peaks_out);
}

//--------------------

} // namespace ImgAlgos

//--------------------
