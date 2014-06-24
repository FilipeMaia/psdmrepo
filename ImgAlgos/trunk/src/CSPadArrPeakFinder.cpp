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
#include <sstream> // for stringstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
// to work with detector data include corresponding 
// header from psddl_psana package
//#include "psddl_psana/acqiris.ddl.h"

#include "PSEvt/EventId.h"
#include "cspad_mod/DataT.h"
#include "cspad_mod/ElementT.h"
#include "ImgAlgos/GlobalMethods.h"

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
  : CSPadBaseModule(name)
  , m_key_signal_out()  
  , m_key_peaks_out()  
  , m_key_peaks_nda()  
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
  m_key_signal_out    = configStr("key_signal_out",        "");
  m_key_peaks_out     = configStr("key_peaks_out",    "peaks");
  m_key_peaks_nda     = configStr("key_peaks_nda",         "");
  m_maskFile_inp      = configStr("hot_pix_mask_inp_file", ""); // "cspad-pix-mask-in.dat"
  m_maskFile_out      = configStr("hot_pix_mask_out_file", "cspad-pix-mask-out.dat");
  m_fracFile_out      = configStr("frac_noisy_evts_file",  "cspad-pix-frac-out.dat");
  m_evtFile_out       = configStr("evt_file_out",          "./cspad-ev-");
  m_rmin              = config   ("rmin",                    3 );
  m_dr                = config   ("dr",                      1 );
  m_SoNThr_noise      = config   ("SoNThr_noise",            3 ); // for noisy pixel counting over frames in m_stat[q][s][r][c] -> mask
  m_SoNThr_signal     = config   ("SoNThr_signal",           5 ); // for signal pixel selection in m_signal[q][s][r][c]
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
  std::fill_n(&m_common_mode[0], int(MaxSectors), float(0));

  setSelectionMode(); // m_sel_mode_str -> enum m_sel_mode 
  resetStatArrays();
  resetSignalArrays();
  getMaskFromFile(); // load the initial hot-pixel mask from file or default 
  if( m_print_bits & 2 ) printInitialMaskStatistics();
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
  printBaseParameters();

  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source                : " << sourceConfigured()
        << "\n key                   : " << inputKey()
        << "\n m_key_signal_out      : " << m_key_signal_out
        << "\n m_key_peaks_out       : " << m_key_peaks_out 
        << "\n m_key_peaks_nda       : " << m_key_peaks_nda 
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

/// Method which is called once at the beginning of the job
void 
CSPadArrPeakFinder::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();

  evaluateVectorOfIndexesForMedian();
  if( m_print_bits & 64 ) printVectorOfIndexesForMedian();
  if( m_print_bits & 64 ) printMatrixOfIndexesForMedian();

  m_time = new TimeInterval();

  //omp_init_lock(m_lock); // initialization, The initial state is unlocked
}

/// Method which is called at the beginning of the run
void
CSPadArrPeakFinder::beginRun(Event& evt, Env& env)
{
  // call base class method
  CSPadBaseModule::beginRun(evt, env);

  // fill segments array
  for (int i = 0; i < MaxQuads; ++i) {
    makeVectorOfSectorAndIndexInArray(i);
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
  m_time -> startTimeOnce();

  if( m_print_bits &  32 ) printTimeStamp(evt);

  maskUpdateControl();
  resetForEventProcessing();
  if( m_print_bits & 512 ) printSelectionStatisticsByCounter();

  procData(evt);

  makeUnitedPeakVector();

  if( v_peaks.size() > 0 ) {
    savePeaksInEvent(evt);
    savePeaksInEventAsNDArr(evt);
  }

  bool isSelected = eventSelector();

  if( m_print_bits &   4 ) printEventSelectionPars(evt, isSelected);
  if( m_print_bits & 128 ) printVectorOfPeaks();

  if (m_sel_mode == SELECTION_ON  && !isSelected) {skip(); return;}
  if (m_sel_mode == SELECTION_INV &&  isSelected) {skip(); return;}

  ++ m_count_selected;
  doOperationsForSelectedEvent(evt);
  if( m_print_bits & 1024 ) printTimeStamp(evt,std::string(" selected"));

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
  //if( m_print_bits & 1024 ) 
  printJobSummary();  
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
CSPadArrPeakFinder::procStatArrays()
{
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
                m_mask[iq][is][ic][ir] = 0; 
                npix_noisy ++;
	      }
	      
              //if (stat > 0) cout << "q,s,c,r=" << iq << " " << is << " " << ic << " " << ir
	      //                 << " stat, total=" << stat << " " << m_count << endl;

            } 
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
/// Reset for event
void
CSPadArrPeakFinder::resetForEventProcessing()
{
  v_peaks.clear(); // clear the vector of peaks

  // Thread safe operation requires undependent data for sections
  for (int q = 0; q < MaxQuads; ++ q) {
    for (int s = 0; s < MaxSectors; ++ s) {
      v_peaks_in_sect[q][s].clear();
    }
  }

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
/// Reset signal arrays
void
CSPadArrPeakFinder::resetSignalArrays()
{
  std::fill_n(&m_signal     [0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0);
  std::fill_n(&m_bkgd       [0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0);
  std::fill_n(&m_noise      [0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, double(0));
  std::fill_n(&m_proc_status[0][0][0][0], MaxQuads*MaxSectors*NumColumns*NumRows, 0);
}

//--------------------
// Process data in the event()

void 
CSPadArrPeakFinder::procData(Event& evt)
{
  bool fillArr = m_key_signal_out != "";

  shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(source(), inputKey());
  if (data1.get()) {

    MsgLog(name(), debug, "Found CsPad::DataV1 object with address " << source() << " and key \"" << inputKey() << "\"");

    ++ m_count;
    //setCollectionMode();
    shared_ptr<cspad_mod::DataV1> newobj(new cspad_mod::DataV1());
    
    int nQuads = data1->quads_shape()[0];

    for (int iq = 0; iq != nQuads; ++ iq) {

      const CsPad::ElementV1& quad = data1->quads(iq);
      const ndarray<const int16_t, 3>& data = quad.data();

      collectStatInQuad(quad.quad(), data.data());

      if(fillArr) {
        m_newdata = new int16_t[data.size()];  // allocate memory for corrected quad-array 
        fillOutputArr(quad.quad(),m_newdata);
        newobj->append(new cspad_mod::ElementV1(quad, m_newdata, m_common_mode));
        //delete [] m_newdata;
      }
    }    
    if(fillArr) evt.put<Psana::CsPad::DataV1>(newobj, source(), m_key_signal_out); // put newobj in event
  }
  
  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(source(), inputKey());
  if (data2.get()) {

    MsgLog(name(), debug, "Found CsPad::DataV2 object with address " << source() << " and key \"" << inputKey() << "\"");

    ++ m_count;
    //setCollectionMode();
    shared_ptr<cspad_mod::DataV2> newobj(new cspad_mod::DataV2());
    
    int nQuads = data2->quads_shape()[0];

    for (int iq = 0; iq != nQuads; ++ iq) {
      
      const CsPad::ElementV2& quad = data2->quads(iq);
      const ndarray<const int16_t, 3>& data = quad.data();

      collectStatInQuad(quad.quad(), data.data());

      if(fillArr) {
        m_newdata = new int16_t[data.size()];  // allocate memory for corrected quad-array 
        fillOutputArr(quad.quad(),m_newdata);
        newobj->append(new cspad_mod::ElementV2(quad, m_newdata, m_common_mode));  
        //delete [] m_newdata;
      }
    } 
    if(fillArr) evt.put<Psana::CsPad::DataV2>(newobj, source(), m_key_signal_out); // put newobj in event
  }
}

//--------------------
/// 
void 
CSPadArrPeakFinder::fillOutputArr(unsigned quad, int16_t* newdata)
{
  //cout << "fillOutputArr for quad=" << quad << endl;

  int ind_in_arr = 0;
  for (int sect = 0; sect < MaxSectors; ++ sect) {
    if (segMask(quad) & (1 << sect)) {
     
      // beginning of the segment data
      const int16_t* sectData = &m_signal[quad][sect][0][0];
      //const int16_t* sectData = &m_bkgd  [quad][sect][0][0]; // for test
      int16_t*       newData  = newdata  + ind_in_arr*SectorSize;

      // Copy regular array m_signal[MaxQuads][MaxSectors][NumColumns][NumRows] to data-style array. 
      for (int i = 0; i < SectorSize; ++ i) {

        newData[i] = sectData[i];
      }                
      ++ind_in_arr;
    }
  }  
}

//--------------------
/// We need in this stuff in order to isolate data for multithreading.
void 
CSPadArrPeakFinder::makeVectorOfSectorAndIndexInArray(unsigned quad)
{
  TwoIndexes sectAndIndexInArray;
  v_sectAndIndexInArray[quad].clear();

  MsgLog(name(), debug, "quad=" << quad);

  int ind_in_arr = 0;
  for (int sect = 0; sect < MaxSectors; ++ sect) {
    if (segMask(quad) & (1 << sect)) {
      sectAndIndexInArray.i = sect;
      sectAndIndexInArray.j = ind_in_arr++;
      v_sectAndIndexInArray[quad].push_back(sectAndIndexInArray);

      MsgLog(name(), debug,  
                 "     sectAndIndexInArray.i = " << sectAndIndexInArray.i
              << "     sectAndIndexInArray.j = " << sectAndIndexInArray.j
             );

    }
  }
}

//--------------------
/// Collect statistics
/// Loop over all 2x1 sections available in the event 
void 
CSPadArrPeakFinder::collectStatInQuad(unsigned quad, const int16_t* data)
{
  int v_size = v_sectAndIndexInArray[quad].size();

//======================
#pragma omp parallel for
//======================

  //for (int sect = 0; sect < MaxSectors; ++ sect) {
  //  if (segMask(quad) & (1 << sect)) {

  //for( vector<TwoIndexes>::const_iterator p  = v_sectAndIndexInArray[quad].begin();
  //                                        p != v_sectAndIndexInArray[quad].end(); p++ ) {
  //    int sect       = p->i;
  //    int ind_in_arr = p->j;

  for( int ivec=0; ivec < v_size; ++ivec ) { // #pragma does not understand vector's iterator...

      int sect       = v_sectAndIndexInArray[quad][ivec].i;
      int ind_in_arr = v_sectAndIndexInArray[quad][ivec].j;
      const int16_t* sectData = data + ind_in_arr*SectorSize;

      collectStatInSect(quad, sect, sectData);
      findPeaksInSect  (quad, sect);
      //cout << "  quad=" << quad << "  sect=" << sect  << "  ind_in_arr=" << ind_in_arr << endl;
  }
}

//--------------------
void 
CSPadArrPeakFinder::makeUnitedPeakVector()
{
  for (int q = 0; q < MaxQuads; ++ q) {
    for (int s = 0; s < MaxSectors; ++ s) {
      for( vector<Peak>::const_iterator p  = v_peaks_in_sect[q][s].begin();
                                        p != v_peaks_in_sect[q][s].end(); p++ ) {
        v_peaks.push_back(*p);
      }
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
      if ( abs( median.SoN ) > m_SoNThr_noise ) m_stat[quad][sect][ic][ir] ++; 

      // 3) For masked array
      if (m_mask[quad][sect][ic][ir] != 0) {

        // 3a) produce signal/background/noise arrays
        m_signal[quad][sect][ic][ir] = int16_t (median.sig); // for signal
        m_bkgd  [quad][sect][ic][ir] = int16_t (median.avg); // for background
        m_noise [quad][sect][ic][ir] =          median.rms;  // for noise

        // 3b) Mark signal pixel for processing
        m_proc_status[quad][sect][ic][ir] = ( median.SoN > m_SoNThr_signal ) ? 255 : 0; 
      }
      else
      {
        m_signal     [quad][sect][ic][ir] = 0;
        m_bkgd       [quad][sect][ic][ir] = 0;
        m_noise      [quad][sect][ic][ir] = 0;
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

	//cout << "Begin the new cluster, q,s,c,r=" << quad << " " << sect << " "  << ic << " " << ir << endl;

	// Initialization of the peak parameters
        PeakWork pw;
	pw.peak_npix       = 0;
	pw.peak_bkgd_tot   = 0;
	pw.peak_noise2_tot = 0;
	pw.peak_amp_tot    = 0;
	pw.peak_amp_max    = 0;
	pw.peak_amp_x_col1 = 0;
	pw.peak_amp_x_col2 = 0;
	pw.peak_amp_x_row1 = 0;
	pw.peak_amp_x_row2 = 0;

	// Begin to iterate over connected region
        // when it is done the connected region is formed
        iterateOverConnectedPixels(quad,sect,ic,ir,pw); 

	if( peakSelector(pw) ) savePeakInVector(quad,sect,pw);
      }
    }
  }
}

//--------------------
// Flood-fill recursive iteration method in order to find the region of connected pixels
void 
CSPadArrPeakFinder::iterateOverConnectedPixels(int quad, int sect, int ic, int ir, PeakWork& pw)
{
  int16_t amp = m_signal[quad][sect][ic][ir];
  double noise= m_noise [quad][sect][ic][ir];

  pw.peak_npix       ++;
  pw.peak_bkgd_tot   +=  m_bkgd[quad][sect][ic][ir];
  pw.peak_noise2_tot +=  noise*noise; // sum the pixel noise quadratically, as randomly fluctuating.
  pw.peak_amp_tot    +=  amp;
  pw.peak_amp_x_col1 += (amp*ic);
  pw.peak_amp_x_col2 += (amp*ic*ic);
  pw.peak_amp_x_row1 += (amp*ir);
  pw.peak_amp_x_row2 += (amp*ir*ir);

  if (amp > pw.peak_amp_max) pw.peak_amp_max = amp;

  m_proc_status[quad][sect][ic][ir] ^= 1; // set the 1st bit to zero.

  if(ir+1 < NumRows    && m_proc_status[quad][sect][ic][ir+1] & 1 ) iterateOverConnectedPixels(quad, sect, ic,   ir+1, pw); 
  if(ic+1 < NumColumns && m_proc_status[quad][sect][ic+1][ir] & 1 ) iterateOverConnectedPixels(quad, sect, ic+1, ir  , pw); 
  if(ir-1 >=0          && m_proc_status[quad][sect][ic][ir-1] & 1 ) iterateOverConnectedPixels(quad, sect, ic,   ir-1, pw); 
  if(ic-1 >=0          && m_proc_status[quad][sect][ic-1][ir] & 1 ) iterateOverConnectedPixels(quad, sect, ic-1, ir  , pw); 
}


//--------------------
// Check the peak quality and return true for good peak
bool
CSPadArrPeakFinder::peakSelector(PeakWork& pw) {

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
CSPadArrPeakFinder::printPeakWork(PeakWork& pw) {

    double col    = pw.peak_amp_x_col1 / pw.peak_amp_tot;
    double row    = pw.peak_amp_x_row1 / pw.peak_amp_tot; 
    pw.peak_noise = std::sqrt( pw.peak_noise2_tot / pw.peak_npix );
    pw.peak_SoN   = (pw.peak_noise > 0) ? pw.peak_amp_tot / pw.peak_noise : 0;

    MsgLog(name(), info, 
             "Peak candidate:" 
          << "q/s/c/r="           << m_quad
          << "/"                  << m_sect << std::setprecision(1) << std::fixed
	  << "/"                  << col
	  << "/"                  << row
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
CSPadArrPeakFinder::savePeakInVector(int quad, int sect, PeakWork& pw) {
  //MsgLog(name(), info, "Save peak info, npix =" << m_peak_npix << ", amp=" << m_peak_amp_tot;);
  Peak peak;
  peak.quad      = quad;
  peak.sect      = sect;
  peak.col       = pw.peak_amp_x_col1 / pw.peak_amp_tot;
  peak.row       = pw.peak_amp_x_row1 / pw.peak_amp_tot; 
  peak.sigma_col = std::sqrt( pw.peak_amp_x_col2/pw.peak_amp_tot - peak.col*peak.col );
  peak.sigma_row = std::sqrt( pw.peak_amp_x_row2/pw.peak_amp_tot - peak.row*peak.row );
  peak.ampmax    = pw.peak_amp_max;
  peak.amptot    = pw.peak_amp_tot;
  peak.bkgdtot   = pw.peak_bkgd_tot;
  peak.noise     = pw.peak_noise;
  peak.SoN       = pw.peak_SoN;
  peak.npix      = pw.peak_npix;

  //omp_set_lock(m_lock);
  //v_peaks.push_back(peak);   
  //omp_unset_lock(m_lock);
  v_peaks_in_sect[quad][sect].push_back(peak);   
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
    MsgLog(name(), info, 
             "  peak:"      << ++i 
          << "  quad="      << p->quad
          << "  sect="      << p->sect
          << "  col="       << p->col
          << "  row="       << p->row
          << "  npix="      << p->npix 
          << "  SoN="       << p->SoN
          << "  amptot="    << p->amptot
          << "  noise="     << p->noise
          << "  bkgdtot="   << p->bkgdtot
          << "  ampmax="    << p->ampmax
          << "  sigma_col=" << p->sigma_col
          << "  sigma_row=" << p->sigma_row
	   );
  }
}

//--------------------
// Check the peak quality and return true for good peak
bool
CSPadArrPeakFinder::eventSelector() {

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
CSPadArrPeakFinder::printEventSelectionPars(Event& evt, bool isSelected)
{
    MsgLog(name(), info, 
	   //"  Run="         << strRunNumber(evt) 
              "  Event="       << strEventCounter()  
           << "  "             << strTimeStamp(evt)
           << "  mode="        << m_sel_mode_str  
           << "  N peaks/min=" << v_peaks.size()
           << " / "            << m_event_npeak_min
           << "  A tot/thr="   << m_event_amp_tot
           << " / "            << m_event_amp_tot_thr
           << "  isSelected="  << isSelected  
	   );
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
  //res.sig = sectData[row + col*NumRows] - res.avg;    // Signal above the background
    double dat = sectData[row + col*NumRows];
    double dif = dat - res.avg;
    res.sig = (dat>dif) ? dif : dat;                    // Signal above the background
    if (res.rms>0) res.SoN = res.sig/res.rms;           // S/N ratio
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
CSPadArrPeakFinder::printTimeStamp(Event& evt, std::string comment)
{
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {

    MsgLog( name(), info, "Run="    <<  eventId->run()
                       << " Event=" <<  m_count 
                       << " Time="  <<  eventId->time()
	               << comment.c_str() );
  }
}

//--------------------

void
CSPadArrPeakFinder::getMaskFromFile()
{
  if (m_maskFile_inp != std::string("")) {
     MsgLog( name(), info, " Use initial hot pixel mask from file:" << m_maskFile_inp.c_str() ); 
     m_mask_initial = new ImgAlgos::CSPadMaskV1(m_maskFile_inp);
     CSPadMaskV1::mask_t* mask_initial = m_mask_initial->getMask();
     int16_t* mask = &m_mask[0][0][0][0];
     for (int i=0; i<SIZE_OF_ARRAY; i++) mask[i] = mask_initial[i];
  }
  else
  {
     MsgLog( name(), info, " Use default initial hot pixel mask made of units." ); 
     m_mask_initial = new ImgAlgos::CSPadMaskV1(1);
     std::fill_n(&m_mask[0][0][0][0], (int)SIZE_OF_ARRAY, 1);
  }
}

//--------------------

void 
CSPadArrPeakFinder::printInitialMaskStatistics()
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
CSPadArrPeakFinder::doOperationsForSelectedEvent(Event& evt)
{
  // Define the file name
  std::string fname = m_evtFile_out 
                    + strEventCounter() 
                    + "-" + strRunNumber(evt) 
                    + "-" + strTimeStamp(evt);
  std::string fname_arr   =  fname + ".txt";
  std::string fname_peaks =  fname + "-peaks.txt";

  if( m_out_file_bits & 4 ) saveCSPadArrayInFile<int16_t> (fname_arr, m_signal);
  if( m_out_file_bits & 8 ) savePeaksInFile (fname_peaks, v_peaks);
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
CSPadArrPeakFinder::savePeaksInFile (std::string& fname, std::vector<Peak> peaks)
{
  ofstream file; 
  file.open(fname.c_str(),ios_base::out);

  int i=0;
  for( vector<Peak>::const_iterator itv  = peaks.begin();
                                    itv != peaks.end(); itv++ ) {

    if( m_print_bits & 256 ) 
      MsgLog( name(), info, "  peak:"      << ++i
	                 << "  quad="      << itv->quad     
                         << "  segm="      << itv->sect     
                         << "  col="       << itv->col
                         << "  row="       << itv->row
                         << "  npix="      << itv->npix
                         << "  sigma_col=" << itv->sigma_col
                         << "  sigma_row=" << itv->sigma_row
                         << "  ampmax="    << itv->ampmax
                         << "  amptot="    << itv->amptot )

    file << itv->quad      << "  "
         << itv->sect      << "  "
         << itv->col       << "  "
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
// Save vector of peaks in the event asstd::vector<Peak>
void 
CSPadArrPeakFinder::savePeaksInEvent(Event& evt)
{
  if(m_key_peaks_out.empty()) return;

  shared_ptr< std::vector<Peak> >  sppeaks( new std::vector<Peak>(v_peaks) );
  evt.put(sppeaks, source(), m_key_peaks_out);
}

//--------------------
// Save vector of peaks in the event as ndarray<const float,2>
void 
CSPadArrPeakFinder::savePeaksInEventAsNDArr(Event& evt)
{
  if(m_key_peaks_nda.empty()) return;

  ndarray<float, 2> peaks_nda = make_ndarray<float>(int(v_peaks.size()), 12);

  int i=-1;
  for(vector<Peak>::const_iterator itv  = v_peaks.begin();
                                   itv != v_peaks.end(); itv++) {
    i++;
    peaks_nda[i][0] = float(itv->quad);
    peaks_nda[i][1] = float(itv->sect);
    peaks_nda[i][2] = float(itv->col);
    peaks_nda[i][3] = float(itv->row);
    peaks_nda[i][4] = float(itv->sigma_col);
    peaks_nda[i][5] = float(itv->sigma_row); 
    peaks_nda[i][6] = float(itv->ampmax);
    peaks_nda[i][7] = float(itv->amptot);
    peaks_nda[i][8] = float(itv->bkgdtot); 
    peaks_nda[i][9] = float(itv->noise); 
    peaks_nda[i][10]= float(itv->SoN);
    peaks_nda[i][11]= float(itv->npix);
  }

  save2DArrayInEvent<float>(evt, source(), m_key_peaks_nda, peaks_nda);
}

//--------------------
// Print current selection statistics
void 
CSPadArrPeakFinder::printSelectionStatisticsByCounter() // Event& evt)
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
CSPadArrPeakFinder::printSelectionStatistics() // Event& evt)
{
  float fraction = (m_count > 0) ? 100.*float(m_count_selected) / m_count : 0;
  double dt = m_time -> getCurrentTimeInterval();
  float rate = (dt > 0) ? float(m_count) / dt : 0;

  std::stringstream sc; sc.setf(ios_base::fixed); sc.width(7);  sc.fill(' '); sc << m_count;
  std::stringstream sh; sh.setf(ios_base::fixed); sh.width(7);  sh.fill(' '); sh << m_count_selected;
  std::stringstream sf; sf.setf(ios_base::fixed); sf.width(6);  sf.fill(' '); sf.precision(2); sf << fraction;
  std::stringstream st; st.setf(ios_base::fixed); st.width(11); st.fill(' '); st.precision(3); st << dt;
  std::stringstream sr; sr.setf(ios_base::fixed); sr.width(6);  sr.fill(' '); sr.precision(3); sr << rate;
  std::string s = "  NFrames: " + sc.str()
	        + "  NHits: "   + sh.str() + " ("  + sf.str() + "%)" 
	        + "  Time:"     + st.str() + " sec (" + sr.str() + " fps)";

  MsgLog( name(), info, s );
}

//--------------------
// Print current selection statistics
void 
CSPadArrPeakFinder::printJobSummary()
{
  MsgLog( name(), info, "===== JOB SUMMARY =====" );  
  printSelectionStatistics();
  m_time -> stopTime(m_count);
}

//--------------------

} // namespace ImgAlgos

//--------------------
